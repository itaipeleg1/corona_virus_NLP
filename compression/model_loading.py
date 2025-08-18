from pathlib import Path
import torch
import torch.nn as nn
from models.model_config import model_configs
from .compression_configs import compression_configs

# # assumdict = {
#     "state_dict": model.state_dict(),
#     "lr_rate": lr_rate,
#     "best_acc": best_acc
# }



def load_pt_model(model_key: str, model_name: str, model_class: type, tokenizer_class: type, state_dict: dict, num_labels: int = 5, device: str | None = None):
    #getting (model_dict_path: dict, model_key: str, num_labels: int = 5, device: str | None = None):
    #build model based on config
    
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, num_labels=num_labels)

    model.load_state_dict(state_dict)
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

    return student_model, student_tokenizer, student_model_name, student_model_class, student_tokenizer_class


def load_compressed_models(model_key, model_class, tokenizer_class, model_name, num_labels, device):
    """Load all compressed versions (quantized, pruned, distilled) from saved state_dicts."""

    cfg = compression_configs[model_key]
    tokenizer = tokenizer_class.from_pretrained(model_name)

    models = {}

    for comp_type, path in [
        ("quantization", cfg["quantization_path"]),
        ("pruning", cfg["pruning_path"]),
        ("knowledge_distillation", cfg["knowledge_distillation_path"]),
    ]:
        if path.exists():
            print(f"Loading {comp_type} model from {path}")
            state_dict = torch.load(path, map_location=device)

            model = model_class.from_pretrained(model_name, num_labels=num_labels)
            model.load_state_dict(state_dict)
            model.to(device).eval()
            models[comp_type] = model
        else:
            print(f"⚠️ Warning: No saved {comp_type} model found at {path}")
            models[comp_type] = None

    return models, tokenizer



def load_compressed_models(model_key: str, model_name: str, base_model_name: str, model_class: type, tokenizer_class: type, num_labels: int = 5, device: str | None = None):
    #getting (model_dict_path: dict, model_key: str, num_labels: int = 5, device: str | None = None):
    #build model based on config
    '''compression_configs = {
    "covidbert_HF": {
        "base_path": PROJECT_ROOT / "results/best_models/covidbert_HF_study_augmented_state_dict.pt",
        "quantization_path": PROJECT_ROOT / "compression/saved_compressed/covidbert_HF/covidbert_HF_quantization_model.pt",
        "pruning_path": PROJECT_ROOT / "compression/saved_compressed/covidbert_HF/covidbert_HF_pruning_model.pt",
        "knowledge_distillation_path": PROJECT_ROOT / "compression/saved_compressed/covidbert_HF/covidbert_HF_knowledge_distillation_model.pt"
'''
    """
    Load original + compressed models (quantized, pruned, distilled) if available.
    Returns a dictionary of models keyed by type.
    """
    # tokenizer = tokenizer_class.from_pretrained(base_model_name) #original model's tokenizer (from model_config)
    # model = model_class.from_pretrained(base_model_name, num_labels=num_labels)
    paths = compression_configs[model_key]

    original_model = model_class.from_pretrained(base_model_name, num_labels=num_labels)
    original_model.load_State_dict(state_dict)
    q_model = model_class.from_pretrained(base_model_name, num_labels=num_labels)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, tokenizer




