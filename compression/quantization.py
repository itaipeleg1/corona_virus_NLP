import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic


# post-training dynamic quantization - compressing only linear layers
def quantize_model(model, dtype=torch.qint8):
    print("Original model size:", sum(p.numel() for p in model.parameters()))
    model = model.cpu().eval() #quantization is performed on cpu
    q_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8).eval() 
    print("Quantized model size:", sum(p.numel() for p in q_model.parameters()))
    return q_model
