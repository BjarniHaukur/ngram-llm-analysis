from pathlib import Path

import numpy as np
import requests

try: from .dataset import MemmapDataset
except ImportError: from dataset import MemmapDataset

try: from .tokenizer import load_tokenizer
except ImportError: from tokenizer import load_tokenizer

CHECKPOINT_PATH = Path(__file__).parent.parent.parent / "checkpoints" / "ngram/"
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)


def formatted_ngram_probs(batch_results:list[list[tuple[str,list[float]]]], vocab_size:int)->dict[str,np.ndarray]:  # lol, lmao even
    """Converts a batch of outputs from PySmoothedTrie into a dict of 2d arrays (rules -> batch x vocab)"""
    rule_to_probs = {}

    B, V = len(batch_results), vocab_size  # assumes valid output from PySmoothedTrie

    for i, result in enumerate(batch_results):
        for rule, rule_probs in result:
            rule_to_probs[rule] = rule_to_probs.get(rule, np.zeros((B, V)))  # initialize if not present
            rule_to_probs[rule][i] = rule_probs  # fill in the probabilities
                
    return rule_to_probs

class NGramTrie:
    def __init__(self, server_url:str="http://localhost:8080", max_ngram_length:int=7, root_capacity:int=2**14):
        self.server_url = server_url
        self.max_ngram_length = max_ngram_length
        self.vocab_size = root_capacity

    def run_all_metrics(self, tokens:np.ndarray, model_p:np.ndarray)->dict[str, float]:
        # Get predictions for each sequence in the batch
        unsmoothed_ngram_probs = []
        for token_seq in tokens:
            response = requests.post(
                f"{self.server_url}/unsmoothed_predict",
                json={"history": token_seq.tolist()},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                raise RuntimeError(f"Server error: {response.text}")
            unsmoothed_ngram_probs.append(response.json()["probabilities"])

        # Rest of the processing remains the same
        unsmoothed_all_probs = formatted_ngram_probs(unsmoothed_ngram_probs, self.vocab_size)
        unsmoothed_subgram_probs = {k: v for k, v in unsmoothed_all_probs.items() if set(k) == {"+", "-"}}
        unsmoothed_suffix_probs = {k: v for k, v in unsmoothed_all_probs.items() if set(k) == {"+"}}
        
        smoothed_ngram_probs = []
        for token_seq in tokens:
            response = requests.post(
                f"{self.server_url}/smoothed_predict",
                json={"history": token_seq.tolist()},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                raise RuntimeError(f"Server error: {response.text}")
            smoothed_ngram_probs.append(response.json()["probabilities"])
            
        smoothed_all_probs = formatted_ngram_probs(smoothed_ngram_probs, self.vocab_size)
        smoothed_subgram_probs = {k: v for k, v in smoothed_all_probs.items() if set(k) == {"+", "-"}}
        smoothed_suffix_probs = {k: v for k, v in smoothed_all_probs.items() if set(k) == {"+"}}
        
        return {
            **self.calculate_metrics(unsmoothed_all_probs, model_p, "unsmoothed_all"),
            **self.calculate_metrics(unsmoothed_subgram_probs, model_p, "unsmoothed_subgram"),
            **self.calculate_metrics(unsmoothed_suffix_probs, model_p, "unsmoothed_suffix"),
            **self.calculate_metrics(smoothed_all_probs, model_p, "smoothed_all"),
            **self.calculate_metrics(smoothed_subgram_probs, model_p, "smoothed_subgram"),
            **self.calculate_metrics(smoothed_suffix_probs, model_p, "smoothed_suffix"),
            # ... heatmaps, tables, etc.
        }
    
    def calculate_metrics(self, ngram_probs:dict[str,np.ndarray], model_p:np.ndarray, name:str):
        """Metrics regarding all possible ngram rules, for the currently selected rule set"""

        def top_1(i):
            filtered_probs = {r: p for r, p in ngram_probs.items() if len(r) <= i}
            if len(filtered_probs) == 0: return 0
            
            has_argmax_match = np.any([
                np.argmax(model_p, axis=1) == np.argmax(ngram_p, axis=1)
                for ngram_p in filtered_probs.values()
            ], axis=0)
            return float(np.mean(has_argmax_match))
            
        def variation_distance(i):
            filtered_probs = {r: p for r, p in ngram_probs.items() if len(r) <= i}
            if len(filtered_probs) == 0: return 0
            
            smallest_variational_distances = np.min([
                np.abs(ngram_p - model_p).sum(axis=1)
                for ngram_p in filtered_probs.values()
            ], axis=0)
            return float(np.mean(smallest_variational_distances))
        
        return {
            **{f"top_1/{name}_{i}": top_1(i) for i in range(1, self.max_ngram_length)},
            **{f"variation_distance/{name}_{i}": variation_distance(i) for i in range(1, self.max_ngram_length)}
        }
    
  
if __name__ == "__main__":
    from argparse import ArgumentParser
    from ngram_trie import PySmoothedTrie
    
    parser = ArgumentParser()
    parser.add_argument("dataset_file", type=str)
    parser.add_argument("--tokenizer_name", type=str, default="tokenizer")
    parser.add_argument("--ngram_file", type=str, default="ngram")
    parser.add_argument("--ngram_size", type=int, default=7)
    
    args = parser.parse_args()
    
    print(f"Building trie for {args.dataset_file} with tokenizer {args.tokenizer_name} and saving to {args.ngram_file}")
    
    ds = MemmapDataset(args.dataset_file, args.tokenizer_name, num_tokens=int(1e32)) # i.e. just read all the tokens
    tokens = ds[0].tolist()

    tokenizer = load_tokenizer(args.tokenizer_name)
    root_capacity = tokenizer.get_vocab_size()

    print(f"Instantiating trie")
    trie = PySmoothedTrie(n_gram_max_length=args.ngram_size, root_capacity=root_capacity)
    
    print(f"Fitting trie")
    trie.fit(tokens, n_gram_max_length=args.ngram_size, root_capacity=root_capacity)
    
    print(f"Saving trie to {CHECKPOINT_PATH / args.ngram_file}")
    trie.save(str(CHECKPOINT_PATH / args.ngram_file))
