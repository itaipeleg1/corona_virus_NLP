from pathlib import Path
import torch
import torch.nn as nn
from models.model_config import model_configs
from .compression_configs import compression_configs
import torch

# # assumdict = {
#     "state_dict": model.state_dict(),
#     "lr_rate": lr_rate,
#     "best_acc": best_acc
# }

#improve it...

def load_pt_model(model_key: str, state_dict: dict, num_labels: int = 5, device: str | None = None):
    #getting (model_dict_path: dict, model_key: str, num_labels: int = 5, device: str | None = None):
    #build model based on config
        # Load model configuration
    base_model_name = model_key.split('_')[0] #turn "bertweet_HF" to "bertweet" to access model architecture from model_configs
    config = model_configs[base_model_name]
    model_class = config["model_class"]
    print(f'model class is: {model_class}')
    tokenizer_class = config["tokenizer_class"]
    print(f'tokenizer class is: {tokenizer_class}')
    model_name = config["model_name"]
    tokenizer = tokenizer_class.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name, num_labels=num_labels)
  
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

def load_compressed_models(model_key: str, student_key: str, num_labels: int = 5, device: str | None = None):
    """
    Load original and compressed versions (quantized, pruned, distilled) of a model.
    Uses student_key pointer in model_configs for distilled models.
    """
   
    # Prepare configs
    c_config = compression_configs[model_key]

    #error handling
    for key in ["quantization_path", "pruning_path", "knowledge_distillation_path"]:
        path = c_config.get(key)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(f"Path for {key} does not exist: {path}. Did you train the models?")


    # Quantized model
    q_model, q_tokenizer = load_pt_model(
        model_key=model_key,
        state_dict=torch.load(c_config["quantization_path"], map_location="cpu"),
        num_labels=num_labels,
        device=device
    )

    #pruned model
    p_model, p_tokenizer = load_pt_model(
        model_key=model_key,
        state_dict=torch.load(c_config["pruning_path"], map_location="cpu"),
        num_labels=num_labels,
        device=device
    )

    # Distilled model (student architecture + KD weights)
    distilled_model, distilled_tokenizer = load_student_model(
        student_key=student_key,
        num_labels=num_labels,
        device=device
    )
    distilled_model.eval()

    kd_path = c_config.get("knowledge_distillation_path")
    kd_state_dict = torch.load(kd_path, map_location="cpu")
    distilled_model.load_state_dict(kd_state_dict)
    return {
        "quantized": (q_model, q_tokenizer),
        "pruned": (p_model, p_tokenizer),
        "distilled": (distilled_model, distilled_tokenizer)
    }


