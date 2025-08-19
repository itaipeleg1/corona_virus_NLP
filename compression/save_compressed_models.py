import torch
from config import COMPRESSION_OUTPUT_DIR
from pathlib import Path

def save_model_state(model, model_key, compression_type, output_dir=COMPRESSION_OUTPUT_DIR):
    """
    Saves model state_dict in structured directory:
    compression_outputs/{model_key}/{compression_type}_model.pt
    """
    output_path = Path(output_dir) / model_key
    output_path.mkdir(parents=True, exist_ok=True)

    # State dict only
    state_path = output_path / f"{model_key}_{compression_type}_state_dict.pt"
    torch.save(model.state_dict(), state_path)

    # Full model (including architecture class)
    full_path = output_path / f"{model_key}_{compression_type}_full.pt"
    torch.save(model, full_path)

    print(f"✅ Saved {compression_type} state_dict to {state_path}")
    print(f"✅ Saved {compression_type} full model to {full_path}")

    return state_path, full_path
