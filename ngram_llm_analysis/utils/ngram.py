from pathlib import Path

import torch
import numpy as np
from ngram_trie import PySmoothedTrie

try: from .dataset import MemmapDataset
except ImportError: from dataset import MemmapDataset

CHECKPOINT_PATH = Path(__file__).parent.parent.parent / "checkpoints" / "ngram/"
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

def formatted_ngram_probs(token_probs:list[tuple[int, list[tuple[str, float]]]])->dict[str, np.ndarray]:
    rule_to_probs = {}
    
    for token_idx, rule_probs in token_probs:
        for rule, prob in rule_probs:
            rule_to_probs[rule] = rule_to_probs.get(rule, np.zeros(len(token_probs)))
            rule_to_probs[rule][token_idx] = prob
            
    return rule_to_probs

class NGramTrie:  # wrapping the ngram-trie crate
    def __init__(self, ngram_file:str, max_ngram_length:int=7):
        self.trie = PySmoothedTrie.load(CHECKPOINT_PATH / ngram_file)
        self.max_ngram_length = max_ngram_length
        
    def run_all_metrics(self, text, logits):
        return {
            **self.all_metrics(text, logits), # important to do this first for caching reasons
            **self.subgram_metrics(text, logits),
            **self.suffix_metrics(text, logits),
            # **self.backoff_metrics(text, logits),
            # ... heatmaps, tables, etc.
        }
    
    def calculate_metrics(self, texts:list[str], model_p:np.ndarray, name:str):
        """Metrics regarding all possible ngram rules, for the currently selected rule set"""
                
        probs = [self.trie.get_smoothed_probabilities(text) for text in texts]
        probs = [formatted_ngram_probs(p) for p in probs]
        
        def top_1(i):
            filtered_probs = [{r: p for r, p in rule_dict.items() if len(r) <= i} for rule_dict in probs]
            has_argmax_match = [
                np.any([np.argmax(model_p) == np.argmax(p) for p in rule_dict.values()])
                for rule_dict in filtered_probs
            ]
            return np.mean(has_argmax_match)
            
        def variation_distance(i):
            filtered_probs = [{r: p for r, p in rule_dict.items() if len(r) <= i} for rule_dict in probs]
            smallest_variational_distances = [
                np.min([np.abs(p - model_p).sum() for p in rule_dict.values()])
                for rule_dict in filtered_probs
            ]
            return np.mean(smallest_variational_distances)
        
        return {
            **{f"top_1/{name}_{i}": top_1(i) for i in range(1, self.max_ngram_length)},
            **{f"variation_distance/{name}_{i}": variation_distance(i) for i in range(1, self.max_ngram_length)}
        }
        
    def all_metrics(self, texts:list[str], model_p:np.ndarray):
        self.trie.set_all_ruleset_by_length(self.max_ngram_length-1)  # -1 for context
        return self.calculate_metrics(texts, model_p, "all")

    def subgram_metrics(self, texts:list[str], logits:torch.Tensor):
        self.trie.set_subgram_ruleset_by_length(self.max_ngram_length-1)
        return self.calculate_metrics(texts, logits, "subgram")
    
    def suffix_metrics(self, texts:list[str], logits:torch.Tensor)->dict:
        self.trie.set_suffix_ruleset_by_length(self.max_ngram_length-1)
        return self.calculate_metrics(texts, logits, "suffix")
    
    def backoff_metrics(self, texts:list[str], logits:torch.Tensor)->dict:
        self.trie.set_backoff_ruleset_by_length(self.max_ngram_length-1)
        return self.calculate_metrics(texts, logits, "backoff")
  
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("dataset_file", type=str)
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("ngram_file", type=str)
    
    args = parser.parse_args()
    
    print(f"Building trie for {args.dataset_file} with tokenizer {args.tokenizer_name} and saving to {args.ngram_file}")
    
    ds = MemmapDataset(args.dataset_file, args.tokenizer_name, num_tokens=int(1e32)) # i.e. just read all the tokens
    tokens = ds[0].tolist()
    
    trie = PySmoothedTrie(n_gram_max_length=7)
    trie.fit(tokens, n_gram_max_length=7, root_capacity=max(tokens)+1)
    print(f"Saving trie to {CHECKPOINT_PATH / args.ngram_file}")
    trie.save(str(CHECKPOINT_PATH / args.ngram_file))
    
  
  
  
  
  
  
  
  
  
        
#     def top1_accuracy(self, text, rules, model):
#         '''return {rule: {n_gram: prob, transformer: prob}}'''
        
#         self.trie.set_rule_set(rules)

#         self.trie.fit_smoothing()
        
#         bos_token = tok.token_to_id("<bos>")
#         input_ids = [bos_token] + tok.encode(text, add_special_tokens=False).ids
        
#         n_gram_probs = self.trie.get_prediction_probabilities(input_ids)
        
#         rule_to_ngram_probs = formatted_ngram_probs(n_gram_probs)

#         top1_accuracy = {}
        
#         for rule in rules:
#             top1_match = 0
#             import torch.nn.functional as F
#             logits = model_logits(input_ids, model, rule)
#             p = F.softmax(logits, dim=-1)
#             top_probs, top_indices = torch.topk(p, k=1)
                        
#             # if predicting the same token as the ngram model, increment top1_match
#             ngram_token = rule_to_ngram_probs[rule][0][0]
#             model_token = top_indices[0].item()
            
#             top1_match += ngram_token == model_token
            
#             top1_accuracy[rule] = {"ngram": {"token": ngram_token, "prob": rule_to_ngram_probs[rule][0][1]}, "model": {"token": model_token, "prob": top_probs[0].item()}, "match": ngram_token == model_token}
        
#         matches = sum(top1_accuracy[rule]["match"] for rule in rules)
#         top1_accuracy["accuracy"] = matches / len(rules)
        
#         return top1_accuracy
            
# if __name__ == "__main__":
#     from itertools import product

#     import yaml
    
#     from train import model_from_config
#     from utils.tokenizer import load_tokenizer
        
#     from utils.dataset import MemmapDataset
#     from utils.tokenizer import load_tokenizer
#     from torch.utils.data import random_split

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     tok = load_tokenizer("tokenizer_bytes")

#     with open("../data/small_train.txt", "r") as f:
#         f.readline() # the first line is just "text"
#         data = f.read()
        
#     with open("../configs/llama_medium.yaml", "r") as f:
#         config = yaml.safe_load(f)
        
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#     model = model_from_config(config).to(DEVICE)

#     tok = load_tokenizer("tokenizer_bytes")
#     tok.pad_token = tok.token_to_id("<pad>")
#     full_ds = MemmapDataset(dataset_file="small_train", tokenizer_name='tokenizer_bytes', num_tokens=2048 - 1)

#     train_size = int(0.8 * len(full_ds))
#     val_size = int(0.1 * len(full_ds))
#     test_size = len(full_ds) - train_size - val_size
#     train_ds, val_ds, _ = random_split(full_ds, [train_size, val_size, test_size])

#     tokenized_data = []

#     for batch in train_ds:
#         tokenized_data.extend(batch.tolist())
        
#     ns = NgramStats(tokenized_data)
        
#     #symbols = ['-', '+', '*'] # TODO: add support for marginalization
#     symbols = ['-', '+']
#     rules = []

#     for length in range(1, 7):
#         rules.extend([''.join(p) for p in product(symbols, repeat=length)])
        
#     prompt = "time. in a big"
#     top1 = ns.top1_accuracy(prompt, rules, model)
#     print(top1)
