from utils import load_jsonl
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from tqdm import tqdm

import sys
import re
from collections import Counter
import numpy as np
import gc

import torch
torch.cuda.empty_cache()
torch.set_grad_enabled(False)

print("Cuda is available", torch.cuda.is_available())

from FlagEmbedding import BGEM3FlagModel

def load_data(dname, dir, max_samples):
    data = load_jsonl(dir)

    # neutral
    if dname in ['TST', 'OE']:
        data_out = data[:max_samples]
    else:
        # other
        n_per_tertile = max_samples // 3

        data_by_tertiles = {'low': [], 'medium': [], 'high': []}

        for item in data:
            data_by_tertiles[item['word_tertile']].append(item)

        data_out =  (data_by_tertiles['low'][:n_per_tertile] + 
                data_by_tertiles['medium'][:n_per_tertile] + 
                data_by_tertiles['high'][:n_per_tertile])

    # return hwt, mgt lists
    return [item['trgt'] for item in data_out], [item['mgt'] for item in data_out]
    
def compute_ppl(texts, tokenizer, model, device):
    ppl_scores = []

    for text in tqdm(texts):
        text = text.strip()
        if not text:
            ppl_scores.append(float("nan"))
            continue

        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            input_ids = inputs["input_ids"]

            logits = model(**inputs).logits[:, :-1, :]
            labels = input_ids[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="mean"
            )
            ppl = torch.exp(loss).item()

            torch.cuda.empty_cache()

            ppl_scores.append(ppl)

    # Remove NaNs if any
    ppl_scores = [score for score in ppl_scores if not np.isnan(score)]

    # Compute percentiles
    lower = np.percentile(ppl_scores, 3)
    upper = np.percentile(ppl_scores, 97)

    # Filter out bottom and top 1% ....
    filtered_scores = [score for score in ppl_scores if lower <= score <= upper]

    return filtered_scores


def compute_levenshtein(texts1, texts2):
    assert len(texts1) == len(texts2), "different length"
    
    scores = []
    for text1, text2 in zip(texts1, texts2):
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        edit_distance = levenshtein_distance(text1, text2)
        max_len = max(len(text1), len(text2))
        normalized_distance = edit_distance / max_len
        scores.append(normalized_distance)
    
    return scores


def compute_semantic_similarities(texts1, texts2, emb_model):
    assert len(texts1) == len(texts2), "different length"

    embeddings_1 = emb_model.encode(texts1, 
                                    batch_size=16,
                                    max_length=512,)['dense_vecs']
    embeddings_2 = emb_model.encode(texts2, 
                                    batch_size=16,
                                    max_length=512)['dense_vecs']
    
    # https://github.com/UKPLab/sentence-transformers/issues/822
    # scores = util.dot_score(emb1, emb2)
    similarities = []
    for i in tqdm(range(len(texts1))):
        similarity = util.dot_score(embeddings_1[i], embeddings_2[i]) / (np.linalg.norm(embeddings_1[i]) * np.linalg.norm(embeddings_2[i]))
        similarities.append(similarity)
    
    return similarities

def compute_unigram_overlap(texts1, texts2):
    assert len(texts1) == len(texts2), "check length"
    
    overlaps = []
    
    for text1, text2 in zip(texts1, texts2):
        
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()
        
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)
        
        overlap_tokens = counter1 & counter2
        
        intersection_size = sum(overlap_tokens.values())
        union_size = sum(counter1.values()) + sum(counter2.values()) - intersection_size
        
        overlap_score = intersection_size / union_size
        overlaps.append(overlap_score)
    
    return overlaps

def plot_metric_distributions(dataset_results, metric_name, save_path=None):
    plt.figure(figsize=(12, 8))
    
    color_map = {
    'OE': '#1f77b4',
    'PW': '#ff7f0e',
    'SUM': '#2ca02c',
    'TST': '#d62728',
    'HWT': '#9467bd'
    }

    for i, (dataset_name, scores) in enumerate(dataset_results.items()):
        
        flat_scores = np.ravel(scores)
        if scores == 'ppl':
            print(flat_scores.shape)
        sns.kdeplot(flat_scores, label=f"{dataset_name}", 
                    fill=True, alpha=0.3, color=color_map.get(dataset_name, None))
    
    if metric_name == 'levenshtein':
        plt.xlabel("Normalized Levenshtein Distance", fontsize=14)
    elif metric_name == 'semantic':
        plt.xlabel("Cosine Similarity", fontsize=14)
    elif metric_name == 'unigram':
        plt.xlabel("Unigram Overlap", fontsize=14)
    elif metric_name == 'ppl':
        plt.xlabel("Perplexity", fontsize=14)

    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_combined_metrics(levenshtein_results, semantic_results, unigram_results, pplx_results, save_path=None):

    fig, axes = plt.subplots(1, 4, figsize=(26, 8))
    
    color_map = {
        'OE': '#e41a1c',
        'PW': '#377eb8',
        'SUM': '#4daf4a',
        'TST': '#ff7f00',
        'HWT': '#9932cc'
    }
    
    # Levenshtein
    for i, (dataset_name, scores) in enumerate(levenshtein_results.items()):
        sns.kdeplot(scores, label=dataset_name, fill=True, alpha=0.3, color=color_map.get(dataset_name, None), ax=axes[0])
    axes[0].set_xlabel("Norm. Levenshtein Distance", fontsize=28)
    axes[0].set_ylabel("Density", fontsize=28)
    axes[0].grid(False)
    
    # Semantic
    for i, (dataset_name, scores) in enumerate(semantic_results.items()):
        flat_scores = np.ravel(scores)
        sns.kdeplot(flat_scores, label=dataset_name, fill=True, alpha=0.3, color=color_map.get(dataset_name, None), ax=axes[1])
    axes[1].set_xlabel("Cosine Similarity", fontsize=28)
    axes[1].set_ylabel("")
    axes[1].grid(False)
    
    # Unigram
    for i, (dataset_name, scores) in enumerate(unigram_results.items()):
        sns.kdeplot(scores, label=dataset_name, fill=True, alpha=0.3, color=color_map.get(dataset_name, None), ax=axes[2])
    axes[2].set_xlabel("Unigram Overlap", fontsize=28)
    axes[2].set_ylabel("")
    axes[2].grid(False)

    # PPLX
    for i, (dataset_name, scores) in enumerate(pplx_results.items()):
        sns.kdeplot(scores, label=dataset_name, fill=True, alpha=0.3, color=color_map.get(dataset_name, None), ax=axes[3])
    axes[3].set_xlabel("Perplexity", fontsize=28)
    axes[3].set_ylabel("")
    axes[3].grid(False)
    axes[3].set_xlim(0, 150)

    handles_labels = [ax.get_legend_handles_labels() for ax in axes]
    all_handles = sum((hl[0] for hl in handles_labels), [])
    all_labels = sum((hl[1] for hl in handles_labels), [])
    
    seen = set()
    unique_handles_labels = [(h, l) for h, l in zip(all_handles, all_labels) if not (l in seen or seen.add(l))]
    unique_handles, unique_labels = zip(*unique_handles_labels)

    fig.legend(unique_handles, unique_labels, loc='lower center', bbox_to_anchor=(0.5, -0.08),
            ncol=len(unique_labels), fontsize=28)


    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.tick_params(axis='both', labelsize=24)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main(max_samples=600):
    
    lang = sys.argv[1]
    print(f'Start building plots for {lang}')
    
    base_dir = ""
    data_sources = [
        ('OE', os.path.join(base_dir, f'generalise/data/ds/external/mgt/wiki_{lang}_gpt.jsonl')),
        ('PW', os.path.join(base_dir, f'paragraphs/ds/mgt/{lang}_paras_rag_first_gpt.jsonl')),
        ('SUM', os.path.join(base_dir, f'summaries/ds/{lang}_sums_mgt_few1_gpt.jsonl')),
        ('TST', os.path.join(base_dir, f'tst/ds/mgt/{lang}{("_default" if lang == "en" else "")}_mgt_few5_gpt.jsonl'))
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b1", torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-1b1")
    model.eval()

    levenshtein_results = {}
    semantic_results = {}
    unigram_results = {} 
    pplx_results = {}
    
    for dname, dpath in data_sources:
        print(f"\nProcessing PPL for {dname} dataset...", flush=True)

        src_texts, gen_texts = load_data(dname, dpath, max_samples)
        if dname == "OE":
            ppl_scores = compute_ppl(gen_texts, tokenizer, model, device)
            pplx_results[dname] = ppl_scores

            ppl_scores = compute_ppl(src_texts, tokenizer, model, device)
            pplx_results['HWT'] = ppl_scores
        else:
            ppl_scores = compute_ppl(gen_texts, tokenizer, model, device)
            pplx_results[dname] = ppl_scores

    # free some space
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    emb_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    #
    for dname, dpath in data_sources:
        print(f"\nProcessing Lev., Sem., and Unigrams for {dname} dataset...", flush=True)
        src_texts, gen_texts = load_data(dname, dpath, max_samples)
        
        lev_scores = compute_levenshtein(src_texts, gen_texts)
        levenshtein_results[dname] = lev_scores
        
        sem_scores = compute_semantic_similarities(src_texts, gen_texts, emb_model)
        semantic_results[dname] = sem_scores
        
        unigram_scores = compute_unigram_overlap(src_texts, gen_texts)
        unigram_results[dname] = unigram_scores

    out_dir = 'assets'
    os.makedirs(out_dir, exist_ok=True)
    
    levenshtein_plot_path = os.path.join(out_dir, f"lev_dis_{lang}.png")
    semantic_plot_path = os.path.join(out_dir, f"sem_dis_{lang}.png")
    unigram_plot_path = os.path.join(out_dir, f"unigram_dis_{lang}.png")
    ppl_plot_path = os.path.join(out_dir, f"ppl_dis_{lang}.png")
    combined_plot_path = os.path.join(out_dir, f"combined_metrics_{lang}.pdf")
    
    plot_combined_metrics(levenshtein_results, semantic_results, unigram_results, pplx_results, combined_plot_path)


if __name__ == "__main__":
    main()