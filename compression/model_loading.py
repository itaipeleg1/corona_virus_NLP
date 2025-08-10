from pathlib import Path
import torch
from models.model_config import model_configs

def load_pt_model(model_path: str, model_key: str, num_labels: int = 5, device: str | None = None):
    """
    Load a state_dict saved via:
        best_model_state = model.state_dict()
        torch.save(best_model_state, path)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    p = Path(model_path).expanduser().resolve()

    cfg = model_configs[model_key]
    ModelClass = cfg["model_class"]
    TokenizerClass = cfg["tokenizer_class"]
    base_name = cfg["model_name"]

    # 1) Build base model + tokenizer
    tokenizer = TokenizerClass.from_pretrained(base_name)
    model = ModelClass.from_pretrained(base_name, num_labels=num_labels)

    # 2) Load your bare state_dict
    state = torch.load(p, map_location="cpu")
    if not (isinstance(state, dict) and all(hasattr(v, "shape") for v in state.values())):
        raise ValueError("Expected a bare state_dict. Got something else.")

    # 4) Load strictly (recommended if num_labels matches training)
    model.load_state_dict(state, strict=True)

    model.to(device).eval()
    return model, tokenizer
