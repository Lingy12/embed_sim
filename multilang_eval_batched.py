import json
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List
import numpy as np
from pathlib import Path
import fire
from metrics.lang_sim import load_data, analyze_similarities, get_output_directory, get_output_filename

def main(data_root: str, model_name: str, 
         subsets: str = "test", lora_path: str = None, batch_size: int = 8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    subsets = subsets.split('+')
    # data = load_data(input_file)
    print(subsets)
    analysis, sample_similarities = analyze_similarities(data_root, model_name, subsets, lora_path, device, batch_size)
    
    output_dir = get_output_directory(model_name, lora_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overall analysis
    output_filename = get_output_filename(model_name, lora_path)
    output_path = output_dir / output_filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    print(f"Overall analysis complete. Results written to {output_path}")
    
    # Save concise summary (average and variance only)
    summary_filename = f"summary_{output_filename}"
    summary_path = output_dir / summary_filename
    summary = {}
    for subset, subset_data in analysis.items():
        summary[subset] = {}
        for lang, lang_data in subset_data.items():
            summary[subset][lang] = {
                "average": lang_data["average"],
                "variance": lang_data["variance"]
            }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1)
    print(f"Concise summary (average and variance) written to {summary_path}")

    # Assuming output_dir is a Path object
    for subset, samples in sample_similarities.items():
        subset_dir = output_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        sample_output_path = subset_dir / "sample_similarities.json"
        with open(sample_output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"Sample-wise similarities for {subset} written to {sample_output_path}")


if __name__ == "__main__":
    fire.Fire(main)