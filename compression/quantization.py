import torch
import copy
from torch.ao.quantization import quantize_dynamic
import torch.nn as nn

def get_model_memory_size(model):
    """Calculate actual memory size in MB"""
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()
    return total_size / (1024 ** 2)

# post-training dynamic quantization - compressing only linear layers
def quantize_model(original_model, dtype=torch.qint8):
    model_copy = copy.deepcopy(original_model)
    model_copy = model_copy.cpu().eval()

    original_size = get_model_memory_size(model_copy)
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Original parameters: {sum(p.numel() for p in model_copy.parameters()):,}")

    q_model = quantize_dynamic(
        model_copy,
        {nn.Linear},  # Specify the layers to quantize
        dtype=dtype,  # Quantization data type
        inplace=False  # Return a new quantized model
    )
    q_model.eval()

    quantized_size = get_model_memory_size(q_model)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Quantized parameters: {sum(p.numel() for p in q_model.parameters()):,}")

    return q_model