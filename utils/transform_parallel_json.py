import json
import fire
from tqdm import tqdm

def transform_parallel_json(input_file, output_file, lang):
    with open(input_file, 'r') as f:
        input_data = json.load(f)
        
    samples = []
    
    for entry in tqdm(input_data):
        samples.append({"src": entry['en'], "targets": {lang: entry[lang]}})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    fire.Fire(transform_parallel_json)