import torch
from config import COMPRESSION_OUTPUT_DIR
from pathlib import Path

def save_model_state(model, model_key, compression_type, output_dir=COMPRESSION_OUTPUT_DIR, summary=None):
    """
    Saves model state_dict in structured directory:
    compression_outputs/{model_key}/{compression_type}_model.pt
    """
    output_path = Path(output_dir) / model_key
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{model_key}_{compression_type}_model.pt"
    torch.save(model.state_dict(), file_path)
    print(f"✅ Saved {compression_type} model to {file_path}")
    
    if summary is not None:
        summary_path = output_path / f"{model_key}_{compression_type}_summary.pt"
        torch.save(summary, summary_path)
        print(f" Saved training summary to {summary_path}")
    return 
