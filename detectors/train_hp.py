import os
import json
import argparse
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)
import evaluate
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from utils import load_jsonl, save_jsonl
import gc

assert torch.cuda.is_available(), "CUDA is not available."
torch.cuda.empty_cache()

def clear_memory():
    print(f"Current allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Memory cleared. Current allocation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

def gen_ds(args, version=1):
    data = load_jsonl(args.in_file)
    
    texts, labels = [], []
    for item in data:

        if args.task != 'extend':
            trgt_text = item['trgt']    
            mgt_text = item['mgt']
        if args.task == 'extend':
            trgt_text = item['trgt_first'].strip() + ' ' + item['trgt'].strip()
            mgt_text = item['trgt_first'].strip() + ' ' + item['mgt'].strip()
        

        # Human text
        texts.append(trgt_text)
        labels.append(0)
        # Machine
        texts.append(mgt_text)
        labels.append(1)


    # Create ads
    ds = Dataset.from_dict({'texts': texts, 'labels': labels})
    ds = ds.shuffle(seed=args.seed)
    
    # limit by n
    if str(args.n) != "all":
        ds = ds.select(range(int(args.n)))
    
    ds_full = ds.train_test_split(test_size=0.2, seed=args.seed)
    
    dev_test = ds_full['test'].train_test_split(test_size=0.5, seed=args.seed)
    
    ds_full.pop('test')
    ds_full['val'] = dev_test['train']
    ds_full['test'] = dev_test['test']
    
    return ds_full

def train_model(model_name, args, ds_tok):
    print(f"\n======================", flush=True)
    print(f"Training model {model_name}...", flush=True)
    
    def model_init():
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            num_labels=2
        )
        model.config.output_attentions = False
        model.gradient_checkpointing_enable()
        return model

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./temp_results",
        save_strategy="no", 
        logging_strategy="no",
        eval_strategy="no",
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        metric_for_best_model="accuracy",
        report_to='none',
        seed=args.seed,
        data_seed=args.seed,
        weight_decay=args.weight_decay,
        fp16=True,
        warmup_steps=args.warmup
    )


    # Metrics
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = acc_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        return {
            "accuracy": accuracy["accuracy"],
            "f1": f1["f1"]
        }

    # Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=ds_tok['train'],
        eval_dataset=ds_tok['val'], #ds_tok['val'] if 'val' in ds_tok.keys() else ds_tok['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    print('Start training...', flush=True)
    trainer.train()
    val_results = trainer.evaluate()

    # test set
    test_results = trainer.evaluate(eval_dataset=ds_tok['test'])

    results = {
        "start_date": args.date, # same start date is the same run
        "name": model_name,
        "in_file": args.in_file,
        "val_accuracy": val_results["eval_accuracy"],
        "val_f1": val_results["eval_f1"],
        "test_accuracy": val_results["eval_accuracy"],
        "test_f1": val_results["eval_f1"],
        "seed": args.seed,
        "train_n": ds_tok['train'].num_rows,
        "val_n": ds_tok['val'].num_rows,
        "test_n": ds_tok['test'].num_rows,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    
    return results

def save_jsonl_append(data, filename):
    try:
        with open(filename, 'r') as f:
            existing_data = [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    with open(filename, 'w') as f:
        for item in existing_data + data:
            f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--models", type=str, nargs='+', default=["microsoft/mdeberta-v3-base", "FacebookAI/xlm-roberta-base"], 
                        help="List of model names to train sequentially")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--date", required=True, type=str, help="Date for unique ids")
    parser.add_argument('--save', action='store_false', help='Enable the feature')
    
    # HPs 
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--n", type=str, default="all", help="'all' or a number")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0, help="Warmup steps.")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    print('\n======================', flush=True)
    print('Load and prep data...', flush=True)
    ds = gen_ds(args)
    print('Total data size: ', sum(len(ds[split]) for split in ds), flush=True)
    
    all_results = {}
    
    # Train each model
    for model_name in args.models:
        model_key = "mdeberta" if "mdeberta" in model_name.lower() else "roberta"
        
        global tokenizer, data_collator
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def tokenize_function(batch):
            return tokenizer(
                batch['texts'], 
                truncation=True, 
                padding=False, 
                max_length=512,
                #return_tensors="pt"
            )
            
        ds_tok = ds.map(tokenize_function, batched=True)
        
        # Train model
        results = train_model(model_name, args, ds_tok)
        
        # Add to combined results
        all_results[model_key] = results
        
        # Clear memory
        clear_memory()
    
    # Save combined results
        out_file_name = args.out_file.replace(".jsonl", f"_{model_key}.jsonl")
        save_jsonl_append([results], out_file_name)
        print(results)
        print(f"\n======================", flush=True)
        print(f"Results saved to {out_file_name}", flush=True)
        
if __name__ == "__main__":
    main()