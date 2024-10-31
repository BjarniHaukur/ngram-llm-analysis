import torch

import numpy as np
def formatted_ngram_probs(token_probs):
    '''
    probs: list(token, list(rule, prob)) 
    returns  {rule: [(token, prob)]}
    
    So the tokens are sorted by their probability given a rule, with NaN values at the bottom
    '''
    
    rule_to_probs = {}
    
    for token, rule_probs in token_probs:
        for rule, prob in rule_probs:
            if rule not in rule_to_probs:
                rule_to_probs[rule] = []
            # Replace NaN with 0
            prob = 0 if np.isnan(prob) else prob
            rule_to_probs[rule].append((token, prob))
    
    # Sort tokens by probability within each rule
    for rule in rule_to_probs:
        rule_to_probs[rule].sort(key=lambda x: x[1], reverse=True)
    
    return rule_to_probs

def apply_rule(context, rule_context):
    '''
    context: list of tokens
    rule: string - "++"
    '''
    for rule in rule_context:
        if rule == "+":
            continue
        elif rule == "-":
            context = context[:-1]
        
    rule_context = rule_context[1:]
    
    return context

    
def model_logits(input_ids, model, rule):
    
    ## TODO: add marginalization
    
    context = apply_rule(input_ids, rule)
    
    if context == []:
        return torch.tensor([0.0] * model.config.vocab_size).unsqueeze(0).to(device)
    
    input_ids_model = torch.tensor(context).unsqueeze(0).to(device)
        
    with torch.no_grad():
        outputs = model(input_ids_model)
        next_token_logits = outputs.logits[:, -1, :]
    
    return next_token_logits
        

from ngram_trie import PySmoothedTrie

class NgramStats:
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data
        

        self.trie = PySmoothedTrie(n_gram_max_length=7, root_capacity=None)

        self.trie.fit(self.tokenized_data, n_gram_max_length=7, root_capacity=None, max_tokens=None)

    def top1_accuracy(self, text, rules, model):
        '''return {rule: {n_gram: prob, transformer: prob}}'''
        
        self.trie.set_rule_set(rules)

        self.trie.fit_smoothing()
        
        bos_token = tok.token_to_id("<bos>")
        input_ids = [bos_token] + tok.encode(text, add_special_tokens=False).ids
        
        n_gram_probs = self.trie.get_prediction_probabilities(input_ids)
        
        rule_to_ngram_probs = formatted_ngram_probs(n_gram_probs)

        top1_accuracy = {}
        
        for rule in rules:
            top1_match = 0
            import torch.nn.functional as F
            logits = model_logits(input_ids, model, rule)
            p = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(p, k=1)
                        
            # if predicting the same token as the ngram model, increment top1_match
            ngram_token = rule_to_ngram_probs[rule][0][0]
            model_token = top_indices[0].item()
            
            top1_match += ngram_token == model_token
            
            top1_accuracy[rule] = {"ngram": {"token": ngram_token, "prob": rule_to_ngram_probs[rule][0][1]}, "model": {"token": model_token, "prob": top_probs[0].item()}, "match": ngram_token == model_token}
        
        matches = sum(top1_accuracy[rule]["match"] for rule in rules)
        top1_accuracy["accuracy"] = matches / len(rules)
        
        return top1_accuracy
            
if __name__ == "__main__":
    import yaml
    import torch
    from utils.tokenizer import load_tokenizer
    from train import model_from_config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer("tokenizer_bytes")

    with open("../data/small_train.txt", "r") as f:
        f.readline() # the first line is just "text"
        data = f.read()
        
    with open("../configs/llama_medium.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_from_config(config).to(DEVICE)
    
    from utils.dataset import MemmapDataset
    from utils.tokenizer import load_tokenizer
    from torch.utils.data import random_split, DataLoader

    tok = load_tokenizer("tokenizer_bytes")
    tok.pad_token = tok.token_to_id("<pad>")
    full_ds = MemmapDataset(dataset_file="small_train", tokenizer_name='tokenizer_bytes', num_tokens=2048 - 1)

    train_size = int(0.8 * len(full_ds))
    val_size = int(0.1 * len(full_ds))
    test_size = len(full_ds) - train_size - val_size
    train_ds, val_ds, _ = random_split(full_ds, [train_size, val_size, test_size])

    tokenized_data = []

    for batch in train_ds:
        tokenized_data.extend(batch.tolist())
        
    ns = NgramStats(tokenized_data)
        
    from itertools import product

    #symbols = ['-', '+', '*'] # TODO: add support for marginalization
    symbols = ['-', '+']
    rules = []

    for length in range(1, 7):
        rules.extend([''.join(p) for p in product(symbols, repeat=length)])
        
    prompt = "time. in a big"
    top1 = ns.top1_accuracy(prompt, rules, model)
    print(top1)