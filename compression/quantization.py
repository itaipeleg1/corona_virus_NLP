import torch
import copy
import torch.quantization
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

def quantize_model_with_wandb(
   original_model,
    model_key: str,
    mode: str = "dynamic",   # "dynamic" or "static"
    dtype: torch.dtype = torch.qint8,
    calibration_loader=None,   # needed for static quantization
    log_to_wandb: bool = True,
):
    """
    Quantizes a model using dynamic or static quantization and logs to wandb.
    """

    model_copy = copy.deepcopy(original_model).cpu().eval()

    original_size = get_model_memory_size(model_copy)
    original_params = sum(p.numel() for p in model_copy.parameters())

    print(f"Original model size: {original_size:.2f} MB")
    print(f"Original parameters: {original_params:,}")

    if mode == "dynamic":
        # don't quantize last linear layer
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

   
    elif mode == "static":
        if calibration_loader is None:
            raise ValueError("Static quantization requires a calibration_loader")

        # Define quantization config
        model_copy.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        # Fuse modules (conv+bn+relu, etc.)
        model_copy_fused = torch.quantization.fuse_modules(
            model_copy,
            [["conv", "bn", "relu"]] if hasattr(model_copy, "conv") else []
        )
        # Prepare
        prepared = torch.quantization.prepare(model_copy_fused)

        # Calibration step
        print("Calibrating...")
        with torch.no_grad():
            for batch in calibration_loader:
                inputs = batch["input_ids"]
                mask = batch["attention_mask"]
                prepared(inputs, mask)

        # Convert to quantized
        q_model = torch.quantization.convert(prepared)

    else:
        raise ValueError("mode must be 'dynamic' or 'static'")

    # ----------------- Metrics ----------------- #
    quantized_size = get_model_memory_size(q_model)
    quantized_params = sum(p.numel() for p in q_model.parameters())
    compression_ratio = original_size / quantized_size

    print(f"{mode.title()} Quantized size: {quantized_size:.2f} MB")
    print(f"Quantized parameters: {quantized_params:,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    # ----------------- Save ----------------- #
    output_path = Path(output_dir) / model_key
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f"{model_key}_{mode}_quantized.pt"
    torch.save(q_model.state_dict(), save_path)

    # ----------------- WandB Logging ----------------- #
    if log_to_wandb:
        wandb.log({
            f"{mode}/model_size_MB": quantized_size,
            f"{mode}/params": quantized_params,
            f"{mode}/compression_ratio": compression_ratio,
        })

    return q_model