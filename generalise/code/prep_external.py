import os
import json
import random
from datasets import load_dataset, Dataset
from utils import save_jsonl, load_jsonl
import random
import regex as rex
import pandas as pd

data_dir = "generalise/data/ds/external"
n=2700

# =======================================================
# Portuguese
# =======================================================

def pt_news():
    ds = load_dataset("emdemor/news-of-the-brazilian-newspaper")
    print(f"PT News\n\n")
    
    train_ds = ds["train"]
    shuffled_ds = train_ds.shuffle(seed=42)
    sampled_ds = shuffled_ds.select(range(n))

    def rename_and_filter(example):
        return {"title": example["title"], "trgt": example["text"]}

    processed_ds = sampled_ds.map(rename_and_filter, remove_columns=[col for col in sampled_ds.column_names if col not in ["title"]])
    print(processed_ds[0].keys(), processed_ds[0])
    save_jsonl(processed_ds, os.path.join(data_dir, "news_pt.jsonl"))
    print('\nPT news new saved', len(processed_ds))


def pt_reviews():
    csv_path = "generalise/data/ds/external/src/B2W-Reviews01.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"PT Reviews\n\n")

    df = df[["review_title", "review_text"]].rename(columns={
        "review_title": "title",
        "review_text": "trgt"
    })

    ds = Dataset.from_pandas(df)
    shuffled_ds = ds.shuffle(seed=42)
    sampled_ds = shuffled_ds.select(range(n))

    print(sampled_ds[0].keys(), sampled_ds[0])
    save_jsonl(sampled_ds, os.path.join(data_dir, "reviews_pt.jsonl"))
    print('\nPT reviews saved', len(sampled_ds))

# =======================================================
# Vietnamese
# =======================================================

def vi_reviews():

    with open("generalise/data/ds/external/src/1-VLSP2018-SA-Restaurant-train.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = [
        {"trgt": line.strip()}
        for line in lines
        if line.strip() and not line.startswith("#") and not line.startswith("{")
    ]

    # drop short ones
    print('Len before dropping short VI reviews ', len(cleaned_lines))
    cleaned_lines = [x for x in cleaned_lines if len(x['trgt'].split()) > 5]
    print('Len after dropping short VI reviews ', len(cleaned_lines))

    sampled = random.sample(cleaned_lines, n)

    def add_beginning(example):
        example["beginning"] = " ".join(example["trgt"].split()[:10])
        return example

    sampled = [add_beginning(ex) for ex in sampled]

    save_jsonl(sampled, os.path.join(data_dir, "reviews_vi.jsonl"))
    print('\nVI reviews saved', len(sampled))

def vi_news():

    with open("generalise/data/ds/external/src/news_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = [
        {"title": item["title"], "trgt": item["content"]}
        for item in data
        if "title" in item and "content" in item
    ]

    sampled = random.sample(filtered, n)

    save_jsonl(sampled, os.path.join(data_dir, "news_vi.jsonl"))
    print('\nVI news saved', len(sampled))

# =======================================================
# ENGLISH
# =======================================================
def yelp():
    ds = load_dataset("Yelp/yelp_review_full")
    print(f"yelp \n\n")

    train_ds = ds["train"]
    shuffled_ds = train_ds.shuffle(seed=42)
    sampled_ds = shuffled_ds.select(range(n))

    def rename_and_filter(example):
        return {"trgt": example["text"]}

    processed_ds = sampled_ds.map(rename_and_filter, remove_columns=["label", "text"])

    def add_beginning(example):
        example["beginning"] = " ".join(example["trgt"].split()[:10])
        return example

    processed_ds = processed_ds.map(add_beginning)
    print(processed_ds[0].keys(), processed_ds[0])
    save_jsonl(processed_ds, os.path.join(data_dir, "yelp_en.jsonl"))

def dm_cnn():
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0")
    print(f"dm/cnn\n\n")
    train_ds = ds["train"]
    shuffled_ds = train_ds.shuffle(seed=42)
    sampled_ds = shuffled_ds.select(range(n))

    def rename_and_filter(example):
        return {"trgt": example["article"], "highlights": example["highlights"]}

    processed_ds = sampled_ds.map(rename_and_filter, remove_columns=["article", "highlights", "id"])
    print(processed_ds[0].keys(), processed_ds[0])

    save_jsonl(processed_ds, os.path.join(data_dir, "cnndm_en.jsonl"))

def arxiv():
    ds = load_dataset("gfissore/arxiv-abstracts-2021")
    print(f"arxiv\n\n")
    train_ds = ds["train"]
    shuffled_ds = train_ds.shuffle(seed=42)
    sampled_ds = shuffled_ds.select(range(n))

    def rename_and_filter(example):
        return {"title": example["title"], "trgt": example["abstract"]}

    processed_ds = sampled_ds.map(rename_and_filter, remove_columns=["title", "abstract", "id", "submitter", "authors", 
                                                                     "comments", "categories", "versions", "journal-ref",
                                                                     "doi", "report-no"])
    print(processed_ds[0].keys(), processed_ds[0])
        
    save_jsonl(processed_ds, os.path.join(data_dir, "arxiv_en.jsonl"))


def wikipedia():

    def clean(x):
        """Removes references and trims text."""
        refs_PATTERN = r'\[sup:.*?:sup\]'
        x = rex.sub(refs_PATTERN, '', x).strip()
        x = rex.sub(r'([\(\[\{])\s+', r'\1', x)
        x = rex.sub(r'\s+([\)\]\}])', r'\1', x)
        x = rex.sub(r'\s+([.,:;!?])', r'\1', x)
        x = rex.sub(r'[^\S\r\n]{2,}', ' ', x)


        return rex.sub(refs_PATTERN, '', x).strip()


    for lang in ['en', 'pt', 'vi']:

        data_dir = f"aid/data/{lang}" # this needs to be populated first; download the data from HF
        paras_dir = f"paras/data/{lang}/ds"
        sums_dir = f"sums/data/{lang}/ds"
        out_dir = "generalise/data/ds/external"
        
        data = load_jsonl(os.path.join(data_dir, f"3_{lang}_text.jsonl"))
        existing_first = load_jsonl(os.path.join(paras_dir, f"{lang}_paras_context_extend.jsonl"))
        existing_cont = load_jsonl(os.path.join(paras_dir, f"{lang}_paras_context_extend.jsonl"))
        existing_sums =  load_jsonl(os.path.join(sums_dir, f"{lang}_sums_mgt_few1_gpt.jsonl"))

        seen_revids = set()
        
        for item in existing_first:
            if 'revid' in item:
                seen_revids.add(item['revid'])

        for item in existing_cont:
            if 'revid' in item:
                seen_revids.add(item['revid'])

        for item in existing_sums:
            if 'revid' in item:
                seen_revids.add(item['revid'])

        print(f"Total seen revids: {len(seen_revids)}")

        # rm in data
        print('Data N', len(data))
        final_data=[]
        for item in data:
            if (item['revid'] not in seen_revids) and (item['sections'] and item['lead']):
                # format item
                # clean lead and seciotn
                lead = clean('\n\n'.join(item.get('lead', [])))
                body = ""
                for section in item["sections"]:
                    for title, paragraphs in section.items():
                        body += f"{title}\n"
                        body += '\n\n'.join(paragraphs) + '\n\n'
                        # body += "\n\n"
                body = clean(body)

                # clean some keys
                item.pop('lead', None)
                item.pop('sections', None)
                item.pop('refs', None)
                item.pop('infobox', None)
                item.pop('timestamp', None)
                item.pop('pageid', None)
                item.pop('revid', None)

                trgt = lead.strip() +  ' ' + body.strip()
                item['trgt'] = trgt
                final_data.append(item)

        print('Final data N', len(final_data), '\n')
        del data
        sample_wiki = random.sample(final_data, n)
        print(sample_wiki[0].keys(), sample_wiki[0])
        save_jsonl(sample_wiki, os.path.join(out_dir, f"wiki_{lang}.jsonl"))

if __name__ == "__main__":

    random.seed(42)
    
    # PT 
    pt_news()
    pt_reviews()

    # VI 
    vi_news()
    vi_reviews()

    # # EN
    dm_cnn()
    arxiv()
    yelp()

    # All
    wikipedia()