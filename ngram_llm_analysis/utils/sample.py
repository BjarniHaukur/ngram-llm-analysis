from typing import Iterator

import torch
import torch.nn.functional as F

def sample_with_temp(logits:torch.Tensor, temperature:float=1.0)->torch.Tensor:
    p = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(p, 1).squeeze(-1)

def nucleus_sample(logits:torch.Tensor, nucleus_threshold:float=0.9)->torch.Tensor:
    B, D = logits.shape
    p = F.softmax(logits, dim=-1).squeeze()
    sorted_probs, sorted_indices = torch.sort(p, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    nucleus_threshold = torch.tensor([nucleus_threshold]*B, device=logits.device).unsqueeze(1)
    cutoffs = torch.searchsorted(cumulative_probs, nucleus_threshold)
    
    cutoff_indices = cutoffs.expand(-1, D)
    position_indices = torch.arange(D, device=logits.device).unsqueeze(0).expand_as(cutoff_indices)
    
    mask = position_indices <= cutoff_indices
    top_probs = sorted_probs * mask.float()
    top_indices = sorted_indices * mask.long()
    
    top_probs_sum = top_probs.sum(dim=-1, keepdim=True)
    top_probs = torch.where(top_probs_sum > 0, top_probs / top_probs_sum, top_probs)
    
    sampled_indices = torch.multinomial(top_probs, 1).squeeze(-1)

    return torch.gather(top_indices, 1, sampled_indices.unsqueeze(1)).squeeze(-1)

def top_k_sample(logits:torch.Tensor, k:int=50)->torch.Tensor:
    p = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(p, k)
    top_probs /= top_probs.sum()
    sampled_index = torch.multinomial(top_probs, 1)
    return top_indices[sampled_index].squeeze(-1)

@torch.no_grad()
def stream_generation(model, tokenizer, prompt:str="", max_length:int=128, temperature:float=0.7)->Iterator[str]:
    model.eval()
    device = next(model.parameters()).device
    
    bos_token = tokenizer.token_to_id("<bos>")
    input_ids = torch.tensor([bos_token] + tokenizer.encode(prompt, add_special_tokens=False).ids).unsqueeze(0).to(device)
    
    generated = []
    for _ in range(max_length):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        # next_token = nucleus_sample(next_token_logits, 0.5)
        next_token = sample_with_temp(next_token_logits, temperature)
        
        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        
        # Check for end-of-sequence token
        if next_token.item() == tokenizer.token_to_id("<eos>"):
            break
    
        yield tokenizer.decode(next_token.tolist())
    
    model.train()

