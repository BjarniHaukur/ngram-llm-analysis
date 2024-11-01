from pathlib import Path

import wandb
import threading
import numpy as np
from ngram_trie import PySmoothedTrie

try: from .dataset import MemmapDataset
except ImportError: from dataset import MemmapDataset

CHECKPOINT_PATH = Path(__file__).parent.parent.parent / "checkpoints" / "ngram/"
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)


def formatted_ngram_probs(token_probs:list[tuple[str,list[tuple[int,float]]]])->dict[str,np.ndarray]:
    """Convert the output of PySmoothedTrie.get_prediction_probabilities into a dict of 2d arrays (rules -> batch x vocab)"""
    rule_to_probs = {}

    B, V = len(token_probs), len(token_probs[0][1])
    
    for i, (rule, rule_probs) in enumerate(token_probs):
        rule_to_probs[rule] = rule_to_probs.get(rule, np.zeros((B, V)))  # initialize if not present
        rule_to_probs[rule][i] = np.array([prob for _, prob in rule_probs])
            
    return rule_to_probs


class NGramTrie:  # wrapping the ngram-trie crate
    def __init__(self, ngram_file:str, max_ngram_length:int=7, root_capacity:int=2**14):  # 2**14 is the vocab size of the tokenizer
        self.trie = PySmoothedTrie(max_ngram_length, root_capacity)
        self.trie.load(str(CHECKPOINT_PATH / ngram_file))
        self.max_ngram_length = max_ngram_length

    def async_run_all_metrics(self, tokens:np.ndarray, model_p:np.ndarray, step:int):
        def run():
            metrics = self.run_all_metrics(tokens, model_p)
            wandb.log(metrics, step=step)
            
        t = threading.Thread(target=run)
        t.start()

    def run_all_metrics(self, tokens:np.ndarray, model_p:np.ndarray)->dict[str, float]:
        return {
            **self.all_metrics(tokens, model_p), # important to do this first for caching reasons
            **self.subgram_metrics(tokens, model_p),
            **self.suffix_metrics(tokens, model_p),
            # **self.backoff_metrics(tokens, model_p),
            # ... heatmaps, tables, etc.
        }
    
    def calculate_metrics(self, tokens:np.ndarray, model_p:np.ndarray, name:str):
        """Metrics regarding all possible ngram rules, for the currently selected rule set"""
                
        probs = [self.trie.get_prediction_probabilities(tokens) for tokens in tokens]
        probs = [formatted_ngram_probs(p) for p in probs]  # dict -> 2d array instead
        
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
        
    def all_metrics(self, tokens:np.ndarray, model_p:np.ndarray):
        self.trie.set_all_ruleset_by_length(self.max_ngram_length-1)  # -1 for context
        return self.calculate_metrics(tokens, model_p, "all")

    def subgram_metrics(self, tokens:np.ndarray, model_p:np.ndarray):
        self.trie.set_subgram_ruleset_by_length(self.max_ngram_length-1)
        return self.calculate_metrics(tokens, model_p, "subgram")
    
    def suffix_metrics(self, tokens:np.ndarray, model_p:np.ndarray)->dict:
        self.trie.set_suffix_ruleset_by_length(self.max_ngram_length-1)
        return self.calculate_metrics(tokens, model_p, "suffix")
    
    # def backoff_metrics(self, texts:list[str], model_p:np.ndarray)->dict:
    #     self.trie.set_backoff_ruleset_by_length(self.max_ngram_length-1)
    #     return self.calculate_metrics(texts, model_p, "backoff")
    
  
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
    trie = PySmoothedTrie(n_gram_max_length=7, root_capacity=max(tokens)+1)
    
    print(f"Fitting trie")
    trie.fit(tokens, n_gram_max_length=7, root_capacity=max(tokens)+1)
    
    print(f"Saving trie to {CHECKPOINT_PATH / args.ngram_file}")
    trie.save(str(CHECKPOINT_PATH / args.ngram_file))
