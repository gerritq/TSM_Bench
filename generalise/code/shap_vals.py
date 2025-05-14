# https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html#id1

import nlp
import numpy as np
import scipy as sp

from utils import load_jsonl
from datasets import Dataset
import random
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import copy
import torch

n=20

print('N:', n)
print('Available GPUs: ', torch.cuda.device_count())

def load_data(data_dir):
    data = load_jsonl(data_dir)

    texts, labels = [], []
    for item in data:

        texts.append(' '.join(item['trgt'].split()[:160]))
        labels.append(0)

        texts.append(' '.join(item['mgt'].split()[:160]))
        labels.append(1)

    ds = Dataset.from_dict({
    "text": texts,
    "label": labels
        })

    return ds

def shap_values(model_dir, data_dir, model_name=None, data_name=None):

    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import shap
    
    def get_shap_data(ax_obj):
        '''returns shap vals and labels'''

        def filter_sum_bars(bars, yticklabels):
            filtered_bars = []
            filtered_labels = []
            for bar, label in zip(bars, yticklabels):
                if not label.startswith("Sum of"):
                    filtered_bars.append(bar)
                    filtered_labels.append(label)
            
            print('N of bars', len(filtered_bars))
            return filtered_bars, filtered_labels

        bars = ax_obj.patches
        yticklabels = [label.get_text() for label in ax_obj.get_yticklabels()]
        bars, yticklabels = filter_sum_bars(bars, yticklabels)
        values = [bar.get_width() for bar in bars]

        return copy.deepcopy(values), copy.deepcopy(yticklabels)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def f(x, model=model, tokenizer=tokenizer):
        tv = torch.tensor([tokenizer.encode(v, padding="max_length", max_length=500, truncation=True) for v in x]).to(device)
        outputs = model(tv)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
        return val

    data = load_data(data_dir)

    explainer = shap.Explainer(f, tokenizer, seed=42)
    shap_values = explainer(data[-n:], fixed_context=1)

    plt.close() 

    ax = shap.plots.bar(shap_values.abs.max(0), show=False)
    
    return get_shap_data(ax)




def combined_plot(values1, yticklabels1, values2, yticklabels2):

    assert len(values1) == len(values1), "values len do not match"

    assert len(yticklabels1) == len(yticklabels2), "y labs do not match"

    fig, ax = plt.subplots(figsize=(12, 8))

    num_features=5
    top_indices1 = np.argsort([-v for v in values1])[:num_features]
    top_indices2 = np.argsort([-v for v in values2])[:num_features]

    # print('Len indices 1', top_indices1)
    # print(len(values1))
    # print('Len indices 2', top_indices2)

    model1_values = [values1[i] for i in top_indices1]
    model1_labels = [yticklabels1[i] for i in top_indices1]
    model2_values = [values2[i] for i in top_indices2]
    model2_labels = [yticklabels2[i] for i in top_indices2]

    # print(model1_values, model1_labels)
    # print(' ')
    # print(model2_values, model2_labels)
    y_pos = np.arange(num_features * 2)

    combined_values = []
    combined_labels = []
    combined_colors = []

    for i in reversed(range(num_features)):
        combined_values.append(model1_values[i])
        combined_labels.append(model1_labels[i])
        combined_colors.append('#1f77b4')  # Blue for model 1

        combined_values.append(model2_values[i])
        combined_labels.append(model2_labels[i])
        combined_colors.append('#ff7f0e')  # Orange for model 2

    ax.barh(y_pos, combined_values, align='center', color=combined_colors, edgecolor=(1, 1, 1, 0.8))

    for i, (val, y) in enumerate(zip(combined_values, y_pos)):
        ax.text(val + 0.01 * max(combined_values), y, f"{val:.1f}",
                va='center', ha='left', fontsize=18)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined_labels,fontsize=18)

    for i in range(1, num_features):
        ax.axhline(i * 2 - 0.5, color='#888888',  linestyle='-', lw=1.5, zorder=-1)

    legend_elements = [
        Patch(facecolor='#1f77b4', label='Our'),
        Patch(facecolor='#ff7f0e', label='Wiki TST')
    ]
    ax.legend(handles=legend_elements, fontsize=18, loc='lower right')

    ax.set_xlabel('max(|SHAP value|', fontsize=18)
    #ax.set_title('Top 5 Features by Model')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"assets/shap_vals.pdf",
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
        
    values1, yticklabels1 = shap_values("generalise/data/hp/best_first_sums_gpt_en", 
                                        "generalise/data/ds/our/first_sums_en_gpt.jsonl", "m_our", "d_our")

    values2, yticklabels2 =  shap_values("generalise/data/hp/best_wiki_gpt_en", 
                                         "generalise/data/ds/external/mgt/wiki_en_gpt.jsonl", "m_wikioe", "d_wikioe")

    # print(' ')
    # print('Vals the same', values1 == values2)
    # print('y-ticks the same', yticklabels1 == yticklabels2)
    # print(' ')
    # print(values1, yticklabels1)
    # print(' ')
    # print(values2, yticklabels2)

    combined_plot(values1, yticklabels1, values2, yticklabels2)

if __name__ == "__main__":
    main()
