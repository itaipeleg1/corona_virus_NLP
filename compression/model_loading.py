from pathlib import Path
import torch
import torch.nn as nn
from models.model_config import model_configs


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
    config = model_configs[student_key]
    model = config["model_class"].from_pretrained(config["model_name"], num_labels=num_labels)
    tokenizer = config["tokenizer_class"].from_pretrained(config["model_name"])

    model.to(device)

    return model, tokenizer