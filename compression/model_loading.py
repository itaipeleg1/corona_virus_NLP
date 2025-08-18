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

def load_student_model(student_key: str, num_labels: int = 5, device: str | None = None, load_trained: bool = False):
    config = model_configs[student_key]
    model = config["model_class"].from_pretrained(config["model_name"], num_labels=num_labels)
    tokenizer = config["tokenizer_class"].from_pretrained(config["model_name"])

    # load distilled weights if available
    if load_trained and config.get("is_state_dict", False):
        best_path = config["best_path"]
        if best_path.exists():
            state_dict = torch.load(best_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"✅ Loaded distilled weights for {student_key} from {best_path}")
        else:
            print(f"⚠️ No saved distilled weights found at {best_path}, using base pretrained weights.")

    model.to(device)

    return model, tokenizer