import math
from collections import defaultdict

import numpy as np

def bleu_score(predicted_tokens:list[list[int]], label_tokens:list[list[int]], n_gram:int=4):
    """ Largely taken from the d2l.ai book. """
    bleu_scores = []
    for preds, labels in zip(predicted_tokens, label_tokens):
        len_pred, len_label = len(preds), len(labels)
        score = math.exp(min(0, 1 - len_label / len_pred))
        for n in range(1, min(n_gram, len_pred) + 1):
            num_matches, label_subs = 0, defaultdict(int)
            for i in range(len_label - n + 1):
                label_subs[tuple(labels[i: i + n])] += 1
            for i in range(len_pred - n + 1):
                if label_subs[tuple(preds[i: i + n])] > 0:
                    num_matches += 1
                    label_subs[tuple(preds[i: i + n])] -= 1
            score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
        bleu_scores.append(score)
    return np.mean(bleu_scores)


if __name__ == "__main__":    
    pred = [[0, 1, 1, 0]]
    label = [[0, 1, 1, 1]]
    print(f"The BLEU-4 score for {pred} and {label} is {bleu_score(pred, label, 4):.2f}, should be 0")
    print(f"The BLEU-3 score for {pred} and {label} is {bleu_score(pred, label, 3):.2f}, should be around 0.75")
    print(f"The BLEU-2 score for {pred} and {label} is {bleu_score(pred, label, 2):.2f}, should be around 0.75")
    print(f"The BLEU-1 score for {pred} and {label} is {bleu_score(pred, label, 1):.2f}, should be around 0.75")