import yaml
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

def model_from_config(config:dict)->nn.Module:    
    model_type = config.get("model_type", "").lower()
    
    if model_type == "llama":
        model_config = LlamaConfig(**config)
        return LlamaForCausalLM(model_config)
    elif model_type == "gpt2":
        model_config = GPT2Config(**config)
        return GPT2LMHeadModel(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
