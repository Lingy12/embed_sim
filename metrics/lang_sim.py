import json
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List
import numpy as np
from random import sample
import warnings
import os
from config.data_config import data_ratio
from pathlib import Path
import sys
sys.path.append('..')

def load_data(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_list(n):
    step = n // 4
    result = [i * step for i in range(4)]
    result.append(n - 1)
    return result

def get_output_filename(model_name: str, lora_path: str = None) -> str:
    model_base = Path(model_name).name
    if lora_path:
        lora_base = Path(lora_path).stem
        return f"{model_base}_{lora_base}_similarity_analysis.json"
    else:
        return f"{model_base}_similarity_analysis.json"

def get_output_directory(model_name: str, lora_path: str = None) -> Path:
    model_base = Path(model_name).name
    if lora_path:
        lora_base = Path(lora_path).stem
        return Path("log") / f"{model_base}_{lora_base}"
    else:
        return Path("log") / model_base

def compute_layer_similarity(src_hidden: torch.Tensor, tgt_hidden: torch.Tensor, target_layer: int) -> torch.Tensor:
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    src_hidden_layer = src_hidden[target_layer][:, -1, :]
    tgt_hidden_layer = tgt_hidden[target_layer][:, -1, :]
    
    # if torch.isnan(src_hidden).any():
    #     warnings.warn("src_hidden contains NaNs", UserWarning)
    # if torch.isnan(tgt_hidden).any():
    #     warnings.warn("tgt_hidden contains NaNs", UserWarning)
    return cos(torch.nan_to_num(src_hidden_layer), torch.nan_to_num(tgt_hidden_layer))

def calculate_similarities_batch(model, tokenizer, src_sentences: List[str], tgt_sentences: List[str], device: torch.device) -> List[List[float]]:
    src_tokens = tokenizer(src_sentences, return_tensors='pt', padding=True, truncation=True).to(device)
    tgt_tokens = tokenizer(tgt_sentences, return_tensors='pt', padding=True, truncation=True).to(device)
    # print(model(**src_tokens))
    # import pdb;pdb.set_trace()
    with torch.no_grad():
        src_hidden = model(**src_tokens).hidden_states
        tgt_hidden = model(**tgt_tokens).hidden_states
    
    eval_layer_lst = generate_list(model.config.num_hidden_layers)
    similarities = [compute_layer_similarity(src_hidden, tgt_hidden, l) for l in eval_layer_lst]
    return torch.stack(similarities, dim=1).tolist()

def load_model_with_lora(model_name: str, lora_path: str = None, device: torch.device = torch.device("cpu")):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    # tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, torch_dtype=torch.float16).to(device)
    
    if lora_path:
        print("Loading LoRA", lora_path)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, lora_path, adapter_name="Lora")
        model = model.merge_and_unload()
    
    return model, tokenizer

def load_subset(data_root: str, subset: str) -> List[Dict]:
    subset_path = os.path.join(data_root, subset + '.json')
    with open(subset_path, "r") as file:
        data = json.load(file)
    return data

def analyze_similarities(data_root, model_name: str, subsets: List[str], lora_path: str = None, device: torch.device = torch.device("cpu"), batch_size: int = 8) -> Dict:
    model, tokenizer = load_model_with_lora(model_name, lora_path, device)
    
    results = {subset: {lang: [] for lang in load_subset(data_root, subset)[0]["targets"].keys()} for subset in subsets}
    sample_similarities = {subset: [] for subset in subsets}
    
    for subset in subsets:
        data = load_subset(data_root, subset)
        
        # select according to data ratio
        data = sample(data, int(len(data) * data_ratio[subset]))
        
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            src_sentences = [entry["src"] for entry in batch]
            
            for lang in results[subset].keys():
                tgt_sentences = [entry["targets"][lang] for entry in batch]
                batch_similarities = calculate_similarities_batch(model, tokenizer, src_sentences, tgt_sentences, device)
                results[subset][lang].extend(batch_similarities)
            
            for j, entry in enumerate(batch):
                sample_id = entry.get("id", f"{subset}_{i+j}")
                entry["id"] = sample_id
                sample_result = {"id": sample_id, "src": entry["src"], "targets": {}}
                
                for lang in results[subset].keys():
                    sample_result["targets"][lang] = {
                        "sentence": entry["targets"][lang],
                        "average_similarity": np.mean(results[subset][lang][i+j]),
                        "similarities": results[subset][lang][i+j]
                    }
                
                sample_similarities[subset].append(sample_result)
    
    analysis = {}
    for subset, subset_results in results.items():
        analysis[subset] = {}
        for lang, lang_similarities in subset_results.items():
            lang_similarities = np.array(lang_similarities)
            analysis[subset][lang] = {
                "average": np.mean(lang_similarities).tolist(),
                "variance": np.var(lang_similarities).tolist(),
                "layer_averages": np.mean(lang_similarities, axis=0).tolist(),
                "layer_variances": np.var(lang_similarities, axis=0).tolist()
            }
    
    return analysis, sample_similarities