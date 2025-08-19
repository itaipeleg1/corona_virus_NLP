from pathlib import Path
import torch
import torch.nn as nn
from models.model_config import model_configs
from .compression_configs import compression_configs
import torch

def load_pt_model(model_key: str, dict_path: Path, num_labels: int = 5, device: str | None = None):
    #getting (model_dict_path: dict, model_key: str, num_labels: int = 5, device: str | None = None):
    #build model based on config
        # Load model configuration
    base_model_name = model_key.split('_')[0] #turn "bertweet_HF" to "bertweet" to access model architecture from model_configs
    config = model_configs[base_model_name]
    model_class = config["model_class"]
    tokenizer_class = config["tokenizer_class"]
    model_name = config["model_name"]
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, num_labels=num_labels)
  
    checkpoint = torch.load(dict_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print(f'state_dict keys: {state_dict.keys()}')
    else:
        state_dict = checkpoint # this is what it's actually loading in our case
    model.load_state_dict(state_dict) #loading saved state dict for original model
    model.to(device).eval()
    return model, tokenizer

def load_student_model(student_key: str, num_labels: int = 5, device: str | None = None):
    s_config = model_configs[student_key]
    student_model_class = s_config["model_class"]
    student_model_name = s_config["model_name"]
    student_tokenizer_class = s_config["tokenizer_class"]
    student_model = student_model_class.from_pretrained(student_model_name, num_labels=num_labels)
    student_tokenizer = student_tokenizer_class.from_pretrained(student_model_name)

    student_model.to(device)

    return student_model, student_tokenizer 

def load_distilled_model(model_key: str, device: str = 'cpu'):
    """Load distilled model from state dict! uses student architecture)"""
    # Get student config from the base model config
    base_model_name = model_key.split('_')[0]
    student_key = model_configs[base_model_name]["student_key"]  # "distilbert" or "distilbert"
    student_config = model_configs[student_key]
    
    # Create student model architecture
    model = student_config["model_class"].from_pretrained(student_config["model_name"], num_labels=5)
    tokenizer = student_config["tokenizer_class"].from_pretrained(student_config["model_name"])
    
    # Load distilled weights
    distilled_path = compression_configs[model_key]["knowledge_distillation_path"]
    state_dict = torch.load(distilled_path, map_location=device, weights_only=False)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Distilled model loaded from {distilled_path}")
    return model, tokenizer

