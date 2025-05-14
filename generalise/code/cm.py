import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_best_model_accuracy(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    runs = [json.loads(line) for line in lines]
    latest_run = max((x['start_date']) for x in runs)
    all_runs = [x for x in runs if x['start_date'] == latest_run]

    assert len(all_runs) == 12, f"less than 12 HP runs?: {file_path}"        

    best_model = max(all_runs, key=lambda x: x['val_accuracy'])
    return best_model['test_accuracy'] * 100

def get_cross_domain_accuracy(lang, train_dataset, test_dataset, model_type, llm_type="gpt"):
    file_pattern = f"generalise/data/detect/{train_dataset}_2_{test_dataset}_{llm_type}_{lang}.jsonl"
    
    matching_files = glob.glob(file_pattern)
    
    if matching_files:
        file_path = matching_files[0]
        with open(file_path, 'r') as f:
            data = json.loads(f.readline())
            return data.get('accuracy', 0) * 100
    else:
        raise ValueError(f"Warning: Could not find file {file_pattern}")


def populate_model_data(lang, model_type, llm_type):
    if lang == 'en':
        datasets = ["first_sums", "wiki", "cnndm", "yelp"]
        display_datasets = ["Our", "Wiki T2T", "CNN/DM", "Yelp"]
    if lang == 'pt':
        datasets = ["first_sums", "wiki", "news", "reviews"]
        display_datasets = ["Our", "Wiki T2T", "FOLHA", "B2W"]
    if lang == 'vi':
        datasets = ["first_sums", "wiki", "news", "reviews"]
        display_datasets = ["Our", "Wiki T2T", "News25", "ABSA"]
    
    result = {}
    
    for i, train_dataset in enumerate(datasets):
        train_display = display_datasets[i]
        result[train_display] = {}
        
        for j, test_dataset in enumerate(datasets):
            test_display = display_datasets[j]
            
            if train_dataset == test_dataset:
                file_path = f"generalise/data/hp/leader_{train_dataset}_{llm_type.lower()}_{lang}_{model_type.lower()}.jsonl"
                accuracy = get_best_model_accuracy(file_path)
            else:
                accuracy = get_cross_domain_accuracy(lang, train_dataset, test_dataset, model_type, llm_type)
            
            result[train_display][test_display] = accuracy
    
    return result

def convert_to_matrix(data, lang):
    if lang == 'en':
        datasets = ["Our", "Wiki T2T", "CNN/DM", "Yelp"]
    if lang == 'pt':
        datasets = ["Our", "Wiki T2T", "FOLHA", "B2W"]
    if lang == 'vi':
        datasets = ["Our", "Wiki T2T", "News25", "ABSA"]

    matrix = np.zeros((len(datasets), len(datasets)))
    
    for i, train_dataset in enumerate(datasets):
        for j, test_dataset in enumerate(datasets):
            matrix[i, j] = data[train_dataset][test_dataset]

    print(matrix)
    
    return matrix, datasets

def create_confusion_matrix(lang, data, llm_type, model_type, output_dir, output_filename):
    lang_mapping = {'en': 'English', 'pt': 'Portuguese', 'vi': 'Vietnamese'}
    
    matrix, datasets = convert_to_matrix(data, lang)
    
    plt.figure(figsize=(10, 8))
    
    if lang != 'vi':
        ax = sns.heatmap(matrix, annot=True, fmt=".1f",
                        xticklabels=datasets, yticklabels=datasets,
                        annot_kws={"size": 26},
                        cbar=False)
    else:
        ax = sns.heatmap(matrix, annot=True, fmt=".1f",
                xticklabels=datasets, yticklabels=datasets,
                annot_kws={"size": 26})

    if llm_type == 'GPT-4o mini':
        plt.title(f"{lang_mapping[lang]}", fontsize=28)

    if llm_type == 'QWen 2.5':
        plt.xlabel("Test", fontsize=24)
    if lang == 'en':
        plt.ylabel("Train", fontsize=24)
    
    plt.tick_params(axis='both', which='major', labelsize=26)
    
    if lang != 'en':
        ax.set_yticks([])
        ax.set_yticklabels([])

    if llm_type == 'GPT-4o mini':
        ax.set_xticks([])
        ax.set_xticklabels([])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, output_filename)
    #plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return plt

def create_single_model_confusion_plot(llm_type, display_llm):
    langs = ['en', 'pt', 'vi']
    lang_mapping = {'en': 'English', 'pt': 'Portuguese', 'vi': 'Vietnamese'}

    fig, axes = plt.subplots(1, 3, figsize=(24, 8)) 

    for col, lang in enumerate(langs):
        ax = axes[col]
        data = populate_model_data(lang, "mdeberta", llm_type)
        matrix, datasets = convert_to_matrix(data, lang)

        if lang == 'vi':
            heatmap = sns.heatmap(
                matrix,
                annot=True,
                fmt=".1f",
                xticklabels=datasets,
                yticklabels=datasets if lang == 'en' else [],
                annot_kws={"size": 28},
                ax=ax
            )
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=18)
            cbar.set_label("Accuracy", fontsize=22)
        else:
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".1f",
                xticklabels=datasets,
                yticklabels=datasets if lang == 'en' else [],
                annot_kws={"size": 28},
                cbar=False,
                ax=ax
            )
            
        ax.set_title(f"{lang_mapping[lang]}", fontsize=28)
        ax.set_xlabel("Test", fontsize=20)
                
        ax.tick_params(axis='both', labelsize=20)

        if lang == 'en':
            ax.set_ylabel(f"Train", fontsize=20)
        else:
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.set_yticklabels([])

    plt.tight_layout()
    # fig.suptitle(f"{display_llm}", fontsize=30, y=1.05)
    return fig

def main():
    langs=['en', 'pt', 'vi']
    model_types = ["mdeberta"]
    llm_types = ["gpt", "qwen"]
    display_llm_types = ["GPT-4o mini", "Qwen 2.5"]
    
    model_display_name = {'mdeberta': 'mDeBERTa'}
    output_dir = "assets"
    
    # First create individual confusion matrices
    data = {}
    for lang in langs:
        print(f'\nGenerating plots for {lang}')
        for i, llm_type in enumerate(llm_types):
            display_llm = display_llm_types[i]
            data[llm_type] = {}
            for model_type in model_types:
                data[llm_type][model_type] = populate_model_data(lang, model_type, llm_type)
                
                plt = create_confusion_matrix(lang,
                    data[llm_type][model_type], 
                    display_llm, 
                    model_display_name[model_type],
                    output_dir,
                    f"cm_len_{lang}_{llm_type}_{model_type}.pdf"
                )
                
                plt.close()
    
    for i, llm_type in enumerate(llm_types):
        display_llm = display_llm_types[i]
        fig = create_single_model_confusion_plot(llm_type, display_llm)
        fig.savefig(os.path.join(output_dir, f"cm_{llm_type}.pdf"), dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    main()