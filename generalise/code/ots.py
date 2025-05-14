# code taken from the hf model cards
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel,AutoModelForSequenceClassification
import transformers
import torch.nn.functional as F
from model_helpers import inference_model

import numpy as np
from sklearn.metrics import accuracy_score
from utils import load_jsonl, save_jsonl
from sklearn.metrics import roc_curve

def load_data(data_dir):
    random.seed(42)

    data = load_jsonl(data_dir)
    data = random.sample(data, 900)

    texts, labels = [], []
    for item in data:
        texts.append(' '.join(item['trgt'].split()[:160]))
        labels.append(0)
        texts.append(' '.join(item['mgt'].split()[:160]))
        labels.append(1)
    return texts, labels

def predict_single_text(text, model, tokenizer, device, max_len=768, threshold=0.5):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    label = 1 if probability >= threshold else 0
    return probability, label


def desklib_detector(texts, labels):

    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            # Initialize the base transformer model.
            self.model = AutoModel.from_config(config)
            # Define a classifier head.
            self.classifier = nn.Linear(config.hidden_size, 1)
            # Initialize weights (handled by PreTrainedModel)
            self.init_weights()

        def forward(self, input_ids, attention_mask=None, labels=None):
            # Forward pass through the transformer
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask

            # Classifier
            logits = self.classifier(pooled_output)
            loss = None
            if labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.float())

            output = {"logits": logits}
            if loss is not None:
                output["loss"] = loss
            return output

    model_directory = "desklib/ai-text-detector-v1.01"

    # --- Load tokenizer and model ---
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = DesklibAIDetectionModel.from_pretrained(model_directory)

    # --- Set up device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preds = []
    for text in tqdm(texts):
        _, label_pred = predict_single_text(text, model, tokenizer, device)
        preds.append(label_pred)

    accuracy = accuracy_score(labels, preds)

    return accuracy

def RADAR_detector(texts, labels, batch_size=16):
    
    device = "cuda"
    model_name = "TrustSafeAI/RADAR-Vicuna-7B"
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    detector.eval()
    detector.to(device)

    probs = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        with torch.no_grad():
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, 
                              return_tensors="pt")
            inputs = {k:v.to(device) for k,v in inputs.items()}
            batch_output_probs = F.log_softmax(detector(**inputs).logits, -1)[:,0].exp().tolist()
            
        probs.extend(batch_output_probs)

    fpr, tpr, thresholds = roc_curve(labels, probs)
    optimal_idx = (tpr - fpr).argmax() 
    optimal_threshold = thresholds[optimal_idx]

    predictions = [1 if prob >= optimal_threshold else 0 for prob in probs]
    accuracy = accuracy_score(labels, predictions)
    
    return accuracy

def e5_small_detector(texts, labels, batch_size=16):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("MayZhou/e5-small-lora-ai-generated-detector")
    detector = AutoModelForSequenceClassification.from_pretrained("MayZhou/e5-small-lora-ai-generated-detector")

    detector.eval()
    detector.to(device)
    prediction_results = inference_model(detector, texts, nthreads=1)
    
    # Extract the probabilities from the prediction results
    probs = prediction_results.predictions['Predicted_Probs(1)']
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    optimal_idx = (tpr - fpr).argmax()  # Youden's J
    optimal_threshold = thresholds[optimal_idx]
    
    # Convert probabilities to binary predictions using the optimal threshold
    predictions = [1 if prob >= optimal_threshold else 0 for prob in probs]
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    return accuracy


def main():
    data_dirs = [('generalise/data/ds/our/gpt_multi_task_en_gpt.jsonl', 'Our'),
                ('generalise/data/ds/external/mgt/wiki_en_gpt.jsonl', 'Wiki Unc.')]
    
    results = {'desklib': {},
               'radar': {},
               'e5_small': {}}

    for data_dir, data_name in data_dirs:
        texts, labels = load_data(data_dir)
        
        # desklib
        print('     Running desklib ..')
        acc_desklib = desklib_detector(texts, labels)
        results['desklib'][data_name] = acc_desklib

        # radar
        print('     Running radar ..')
        acc_radar = RADAR_detector(texts, labels)
        results['radar'][data_name] = acc_radar

        # radar
        print('     Running e5-small ..')
        acc_e5_small = e5_small_detector(texts, labels)
        results['e5_small'][data_name] = acc_e5_small

    save_jsonl([results], "generalise/data/detect/ex1_ots_trained.jsonl")
    print(results)
    
if __name__ == "__main__":
    main()



