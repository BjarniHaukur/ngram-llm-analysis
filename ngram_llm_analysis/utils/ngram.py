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
                np.min([np.abs(p - model_p).sum() for p in rule_dict.values()])  # abs shouldnt be needed (probabilities are non-negative)
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

    def subgram_metrics(self, texts:list[str], model_p:np.ndarray):
        self.trie.set_subgram_ruleset_by_length(self.max_ngram_length-1)
        return self.calculate_metrics(texts, model_p, "subgram")
    
    def suffix_metrics(self, texts:list[str], model_p:np.ndarray)->dict:
        self.trie.set_suffix_ruleset_by_length(self.max_ngram_length-1)
        return self.calculate_metrics(texts, model_p, "suffix")
    
    def backoff_metrics(self, texts:list[str], model_p:np.ndarray)->dict:
        self.trie.set_backoff_ruleset_by_length(self.max_ngram_length-1)
        return self.calculate_metrics(texts, model_p, "backoff")
    
  
if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("dataset_file", type=str)
    parser.add_argument("--tokenizer_name", type=str, default="tokenizer")
    parser.add_argument("--ngram_file", type=str, default="ngram")
    
    args = parser.parse_args()
    
    print(f"Building trie for {args.dataset_file} with tokenizer {args.tokenizer_name} and saving to {args.ngram_file}")
    
    ds = MemmapDataset(args.dataset_file, args.tokenizer_name, num_tokens=int(1e32)) # i.e. just read all the tokens
    tokens = ds[0].tolist()
    
    print(f"Instantiating trie")
    trie = PySmoothedTrie(n_gram_max_length=7, root_capacity=None)
    
    print(f"Fitting trie")
    trie.fit(tokens, n_gram_max_length=7, root_capacity=max(tokens)+1)
    
    print(f"Saving trie to {CHECKPOINT_PATH / args.ngram_file}")
    trie.save(str(CHECKPOINT_PATH / args.ngram_file))
