from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import Dataset
from utils import save_jsonl, load_jsonl, merge_jsonl
import json
import os
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import evaluate
import random


def compute_test_data(args, 
                    best_model_dir, 
                    test_data,
                    train_name, 
                    test_name, 
                    model_name):

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions)
        }

    def prep_testset(args, test_data):
        
        data = test_data[:args.eval_n]

        formatted_data = []

        for item in data:
            if item:
                if 'trgt' not in item.keys():
                    if 'abstract' in item.keys(): # arxiv
                        trgt_text = item['abstract']
                    if 'article' in item.keys(): # cnn/dm
                        trgt_text = item['article']
                else:
                    trgt_text = item['trgt']
                mgt_text = item['mgt']
                    
                if not trgt_text or not mgt_text:
                    print('Found empty instances')
                    continue
                formatted_data.append({"texts": ' '.join(trgt_text.split()[:160]) , "label": 0})
                formatted_data.append({"texts": ' '.join(mgt_text.split()[:160]) , "label": 1})
        
        assert formatted_data, f"Erro - no data for {test_data}"

        test_ds = Dataset.from_list(formatted_data)
        return test_ds

    set_seed(42)

    # Model     
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    trainer = Trainer(model=model, 
                        data_collator=data_collator,
                        compute_metrics=compute_metrics)

    def tokenize_function(batch):
        return tokenizer(batch['texts'], truncation=True, padding=True, max_length=512)

    # RUN HERE
    ds_test = prep_testset(args, test_data)
    ds_test_tok = ds_test.map(tokenize_function, batched=True)

    eval_results = trainer.evaluate(ds_test_tok)

        
    res = {"training_data": train_name,
            "test_data": test_name,
            "accuracy": eval_results["eval_accuracy"],
            "f1": eval_results["eval_f1"],
            "test_n": len(ds_test_tok)
    }
    print(res)
    
    out_file = os.path.join(args.out_dir, f"{train_name}_2_{test_name}_{model_name}_{args.lang}.jsonl")
    save_jsonl([res], out_file)
    print(f"Evaluation results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--eval_n", type=int, required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    set_seed(42)
    random.seed(42)

    # Testsets
    if args.lang == 'en':
        datasets = ['first_sums', 'wiki', 'cnndm', 'yelp', 'arxiv']
    else: 
        datasets = ['first_sums', 'wiki', 'news', 'reviews']
    base_dir_f = (lambda x: 'generalise/data/ds/our' if 
                x == 'first_sums' else 'generalise/data/ds/external/mgt')

    for ds in datasets:
        #base_dir_train = base_dir_f(ds)
        test_datasets = [x for x in datasets if x != ds] # exclude training

        for model in ['gpt', 'qwen']:

            best_model_dir = f"generalise/data/hp_len/best_{ds}_{model}_{args.lang}"

            for tds in test_datasets:
                base_dir_test = base_dir_f(tds)

                print(f'\n\n {args.lang} RUNNING OOD OF {ds} FOR {tds} FOR MODEL {model}')

                # load test set and reduce to 900
                test_data = load_jsonl(os.path.join(base_dir_test, f"{tds}_{args.lang}_{model}.jsonl"))
                test_data = random.sample(test_data, 900)

                # EVAL
                compute_test_data(args=args, 
                                 best_model_dir=best_model_dir, 
                                 test_data=test_data,
                                 train_name=ds,
                                 test_name=tds,
                                 model_name=model)

if __name__ == "__main__":
    main()