import torch
from torch.ao.quantization import quantize_dynamic
import torch.nn as nn

# post-training dynamic quantization - compressing only linear layers
def quantize_model(original_model, dtype=torch.qint8):
    print("Original model size:", sum(p.numel() for p in original_model.parameters()))
    original_model = original_model.cpu().eval() #quantization is performed on cpu
    q_model = quantize_dynamic(original_model, {nn.Linear}, dtype=torch.qint8)
    q_model = q_model.eval() #quantized model is set to eval
    print("Quantized model size:", sum(p.numel() for p in q_model.parameters()))
    return q_model
