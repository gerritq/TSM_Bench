import argparse
import json
import os
import random
import time
import numpy as np
import torch
import gc
from utils import load_jsonl, save_jsonl
from sklearn.metrics import roc_curve, accuracy_score, f1_score
import multiprocessing

# load detectors
from binoculars import Binoculars
from llr import LLR
from fastdetectgpt import FastDetectGPT
from revise import ReviseDetect
from gecscore import GECScore

print("CPU cores available:", os.cpu_count())
print("Max workers:", multiprocessing.cpu_count())

class LLMDetectionEval:

    def __init__(self, in_file, out_file, task, lang, detectors=None):
                self.in_file = in_file
                self.out_file = out_file
                self.task = task
                self.lang = lang
                self.detectors = detectors
 
    @staticmethod
    def clear_memory():
        print(f"Current allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Memory cleared. Current allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def _get_detector(self, name):
        if name == "binoculars":
            model = self._init_binoculars()
            return {
                "model": model,
                "score_fn": lambda texts: [-x for x in model.compute_score(texts)]
            }
        elif name == "llr":
            model = self._init_llr()
            return {
                "model": model,
                "score_fn": lambda texts: model.compute_llrs(texts)
            }
        elif name == "fastdetectgpt_white":
            model = self._init_fastdetectgpt_white()
            return {
                "model": model,
                "score_fn": lambda texts: model.criterion(texts)
            }
        elif name == "fastdetectgpt_black":
            model = self._init_fastdetectgpt_black()
            return {
                "model": model,
                "score_fn": lambda texts: model.criterion(texts)
            }
        elif name == "revise":
            model = self._init_revise()
            return {
                "model": model,
                "score_fn": lambda texts: model.run(texts)
            }
        elif name == "gecscore":
            model = self._init_gecscore()
            return {
                "model": model,
                "score_fn": lambda texts: model.process_data(texts)
            }
        else:
            raise ValueError(f"Unknown detector: {name}")

    def _init_binoculars(self):
        # OG implementation is "tiiuae/falcon-7b" and "tiiuae/falcon-7b-instruct"
        # but falcon does not support VI nor PT
        return Binoculars(
            observer_name_or_path="Qwen/Qwen2.5-7B",
            performer_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            mode="accuracy"
        )
    
    def _init_llr(self):
        return LLR("bigscience/bloom-3b")
    
    def _init_fastdetectgpt_white(self):
        # Reference == scoring model := white box setting
        reference_model="bigscience/bloom-3b"
        scoring_model="bigscience/bloom-3b"
        return FastDetectGPT(scoring_model=scoring_model, 
                            reference_model=reference_model)

    def _init_fastdetectgpt_black(self):
        # Reference != scoring model := black box setting
        # As in the OG paper, we select the scoring model to be smaller (DetectRL confused them)
        reference_model="bigscience/bloom-3b"
        scoring_model="bigscience/bloom-1b7"
        return FastDetectGPT(scoring_model=scoring_model, 
                            reference_model=reference_model)

    def _init_revise(self):
        # We LLama-3B to reduce costs
        return ReviseDetect(self.lang, 
                            model="gpt-3.5-turbo")

    def _init_gecscore(self):
        # We LLama-3B to reduce costs
        return GECScore(self.lang, 
                        model="gpt-3.5-turbo")

    def _load_data(self):
        '''Loads mgt list, returns singular list with text and label fields'''
        data = load_jsonl(self.in_file)

        out=[]
        for item in data:
            #Filter empty
            if ('trgt' in item.keys() and not item['trgt']) or not item['mgt']:
                print('Found empty instance!')
                continue

            if self.task in ['extend', 'mix']:
                    if ('trgt_first' in item.keys() and not item['trgt_first']):
                        print('Found empty instance!')
                        continue
            
            if self.task == 'train':
                if 'trgt' not in item.keys():
                    if 'abstract' in item.keys(): # arxiv
                        trgt_text = item['abstract']
                    if 'article' in item.keys(): # cnn/dm
                        trgt_text = item['article']
                else:
                    trgt_text = item['trgt']    
                mgt_text = item['mgt']

            if self.task in ['first', 'sums', 'neutral']:
                trgt_text = item['trgt']    
                mgt_text = item['mgt']
                
            if self.task == 'extend':
                trgt_text = item['trgt_first'].strip() + ' ' + item['trgt'].strip()
                mgt_text = item['trgt_first'].strip() + ' ' + item['mgt'].strip()
            

            out.append({'text': trgt_text,
                        'label': 0})

            out.append({'text': mgt_text,
                        'label': 1})

        random.seed(2025)
        random.shuffle(out)

        texts = [x['text'] for x in out]
        labels = [x['label'] for x in out]
        assert len(texts) == len(labels)
        
        return texts, labels

    def _score_metrics(self, predictions):

        labels = np.array([x['label'] for x in predictions])
        preds = np.array([x['pred'] for x in predictions])

        # check for nans
        nan_mask = np.isnan(preds)
        nan_n = np.sum(nan_mask)
        print(f"N nans in predictons: {np.sum(nan_mask)}")   
        if nan_n > 0:
                valid_filter = ~nan_mask
                preds = preds[valid_filter]
                labels = labels[valid_filter]

        fpr, tpr, thresholds = roc_curve(labels, preds)

        #  Youden's J stat
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        preds_classes = [1 if x >= optimal_threshold else 0 for x in preds]

        f1 = float(f1_score(labels, preds_classes))
        acc = float(accuracy_score(labels, preds_classes))

        return f1, acc, optimal_threshold

    def detect(self):

        texts, labels = self._load_data()
        print('\nData Size: ', len(texts), flush=True)
        print('\nExample:', flush=True)
        print(texts[0], labels[0], flush=True)
        print('\n\n', flush=True)
        res={}

        # Requries 80GB: binoculars
        
        if self.detectors:
            detector_names = self.detectors
            print(f'Running eval for selected detectors: {self.detectors}')
        else:
            print('Running eval for *all* detectors')
            detector_names = ["binoculars", "llr", "fastdetectgpt_white", "revise", "gecscore", "fastdetectgpt_black"]
        #detector_names = ["revise"]

        for detector_name in detector_names:
            print(f'\n=====================================', flush=True)
            print(f'Running detection with {detector_name}', flush=True)

            detector = self._get_detector(detector_name)
            print(detector)
            scoring_function = detector['score_fn']
            scores = scoring_function(texts)

            preds=[]
            for i in range(len(texts)):
                preds.append({"text": texts[i],
                              "label": labels[i],
                              "pred": scores[i]
                            })

     
            f1, acc, optimal_threshold = self._score_metrics(preds)
            
            res_detector =  {"name": detector_name,
                                "f1": f1,
                                "accuracy": acc,
                                "optimal_threshold": optimal_threshold
                            }
            res[detector_name] = res_detector
            
            # Free memory
            del detector
            LLMDetectionEval.clear_memory()
            print('Sleeping to ensure space has been freed...', flush=True)
            time.sleep(5)


            save_jsonl([res_detector], self.out_file.replace('.jsonl', f"_{detector_name}.jsonl"))
        return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--detectors", nargs='*', required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()
    
    print(f"\nNumber of available GPUs: {torch.cuda.device_count()}", flush=True)

    detector = LLMDetectionEval(args.in_file, args.out_file, args.task, args.lang, args.detectors)
    results = detector.detect()
    
    print('\n\n', flush=True)
    for detector_name, result in results.items():
        print(f"{detector_name}: F1={result['f1']:.4f}, Accuracy={result['accuracy']:.4f}", flush=True)

if __name__ == "__main__":
    main()