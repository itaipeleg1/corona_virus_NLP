import torch
import copy
from torch.ao.quantization import quantize_dynamic
import torch.nn as nn
from config import COMPRESSION_OUTPUT_DIR
from pathlib import Path
from .compression_configs import compression_configs
from models.model_config import model_configs

def get_model_memory_size(model):
    """Calculate actual memory size in MB"""
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    return total_size / (1024 ** 2)

# post-training dynamic quantization - compressing only linear layers
def quantize_model(original_model, model_key, dtype=torch.qint8, output_dir=COMPRESSION_OUTPUT_DIR):

    model_copy = copy.deepcopy(original_model)
    model_copy = model_copy.cpu().eval()

    original_size = get_model_memory_size(model_copy)
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Original parameters: {sum(p.numel() for p in model_copy.parameters()):,}")

    #don't quantize the last layer:
    last_linear = None
    for name, m in model_copy.named_modules():
            if isinstance(m, nn.Linear):
                last_linear = (name, m)


    q_model = quantize_dynamic(
        model_copy,
        {nn.Linear},  # Specify the layers to quantize
        dtype=dtype  # Quantization data type
    )
    if last_linear is not None:
        name, original = last_linear
        # Putting back the original head
        parent = q_model
        *parents, leaf = name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        setattr(parent, leaf, copy.deepcopy(original).eval())

    quantized_size = get_model_memory_size(q_model)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Quantized parameters: {sum(p.numel() for p in q_model.parameters()):,}")
    return q_model

    # else:
    #     # loading quantized model from path
    #     quant_path = compression_configs[model_key]["quantization_path"]
    #     state_dict = torch.load(quant_path, map_location="cpu")
    #     q_model = copy.deepcopy(original_model)    
    #     q_model.load_state_dict(state_dict)

    #    return q_model.eval()

  
