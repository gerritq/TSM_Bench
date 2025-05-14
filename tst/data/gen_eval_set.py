from utils import load_jsonl, save_jsonl
import random
import sys

lang = sys.argv[1]
subset = sys.argv[2]

os.makedirs("tst/ds/eval", exist_ok=True)

in_file = f'tst/ds/4_{lang}_{subset}.jsonl'
out_file = f'tst/ds/eval/{lang}_{subset}_eval.jsonl'

def main():
    
    data = load_jsonl(in_file)
    print('Total N' , len(data))
    data = [item for item in data if not item['drop']]
    print('Keeping' , len(data), f'of {subset}')
    # draw random sample
    random.seed(42)
    random.shuffle(data)
    data = data[:270]
    
    save_jsonl(data, out_file)

if __name__ == "__main__":
    main()