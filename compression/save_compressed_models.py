import os
import torch

def save_model_state(model, model_key, compression_type, output_root="compression_outputs"):
    """
    Saves model state_dict in structured directory:
    compression_outputs/{model_key}/{compression_type}_model.pt
    """
    output_dir = os.path.join(output_root, model_key)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{compression_type}_model.pt"
    save_path = os.path.join(output_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"📁 Saved {compression_type} model to: {save_path}")
