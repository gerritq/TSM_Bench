import os
import json
import random
from collections import defaultdict, Counter
from utils import load_jsonl, save_jsonl
import sys

def by_model():
    models = ["gpt", "qwen"]
    langs = ["en", "pt", "vi"]
    
    for lang in langs:        
        for model in models:
            out = []
            n= 675
            for i in range(2):
                seen_items = set()
                
                if i == 0:
                    file_path = f"paras/data/{lang}/ds/mgt/{lang}_paras_rag_first_{model}.jsonl"
                else:
                    file_path = f"sums/data/{lang}/ds/{lang}_sums_mgt_few1_{model}.jsonl"
                
                out_dir = f"generalise/data/ds/our/first_sums_{lang}_{model}.jsonl"

                print(f"\n=== Processing {lang} - {('paras' if i == 0 else 'sums')} data with {model} model ===")
                try:
                    data = load_jsonl(file_path)
                    data = [item for item in data if item['word_tertile'] in ['medium', 'high']]
                    if i == 0:
                        task = 'paras'
                    else:
                        task = 'sums'
                    for item in data:
                        item['task']=task
                        item['model']=model

                    for tertile in ['medium', 'high']:
                        data_tertile = [item for item in data if item['word_tertile'] == tertile]
                        data_tertile = [item for item in data_tertile if item['trgt'] not in seen_items]
                        selected_items = data_tertile[:n]
                        out.extend(selected_items)
                        
                        for item in selected_items:
                            seen_items.add(item['trgt'])
                except FileNotFoundError:
                    print(f"File not found: {dir_}")
                except Exception as e:
                    print(f"Error processing {dir_}: {str(e)}")    

            save_jsonl(out, out_dir)
            
if __name__ == "__main__":
    by_model()
