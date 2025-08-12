# comparisons: model_size, inference_time, memory usage and accuracy
import torch
import time
import os
import psutil
import tempfile
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from models.data_preparation import prepare_dataset
from pathlib import Path
import pandas as pd

def evaluate_accuracy(model, eval_dataset, device):
    print(f"Starting accuracy evaluation on {device}")
    print(f"Dataset length: {len(eval_dataset)}")
    model.to(device)
    model.eval()
    preds = []
    labels = []
    
    for i, batch in enumerate(eval_dataset):
        if i % 100 == 0:  # Progress every 100 samples
            print(f"Processing sample {i}/{len(eval_dataset)}")
            
        if i >= 100:  # LIMIT FOR TESTING
            print(f"Stopping at {i} samples for testing")
            break
            
        try:
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            label = batch['labels'].item()

            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(output.logits, dim=-1).item()

            preds.append(pred)
            labels.append(label)
            
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            break

    print(f"Accuracy evaluation complete: {len(preds)} predictions made")
    return accuracy_score(labels, preds)


def get_model_size_in_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / 1024**2
    return total_size_mb

def measure_inference_time(model, tokenizer, eval_dataset, device, runs=100, batch_size=8):
    model.to(device)
    model.eval()

    # Warmup - avoiding inital overheasd by warming up and not measuring
    for _ in range(5):
        batch = [eval_dataset[i] for i in range(batch_size)]
        input_ids = torch.stack([b['input_ids'] for b in batch]).to(device)
        attention_mask = torch.stack([b['attention_mask'] for b in batch]).to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # now evaluate inference time
    start_time = time.time()
    for _ in range(runs):
        batch = [eval_dataset[i] for i in range(batch_size)]
        input_ids = torch.stack([b['input_ids'] for b in batch]).to(device)
        attention_mask = torch.stack([b['attention_mask'] for b in batch]).to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    total_time = time.time() - start_time
    avg_time_ms = (total_time / runs) * 1000
    return round(avg_time_ms, 2)


def measure_gpu_memory(model, tokenizer, sample_text, device='cuda'):
    if device != 'cuda' or not torch.cuda.is_available():
        print("GPU not available. Skipping GPU memory measurement.")
        return None

    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    model.eval()
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    with torch.no_grad():
        _ = model(**inputs)

    mem_used_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    return round(mem_used_mb, 2)

def measure_cpu_memory(model, tokenizer, sample_text, device='cpu'):
    model.to(device)
    model.eval()
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    with torch.no_grad():
        _ = model(**inputs)
    mem_after = process.memory_info().rss
    mem_used_mb = (mem_after - mem_before) / (1024 * 1024)
    return round(mem_used_mb, 2)


def evaluate_model(model, tokenizer, eval_dataset, sample_text, device: str):
     #make sure q_model is on cpu so that it doesn't break
    if isinstance(model, torch.nn.quantized.dynamic.Linear) or 'Quantized' in model.__class__.__name__:
        print("Quantized model detected. Forcing evaluation on CPU.")
        device = 'cpu'

    return {
        "accuracy": evaluate_accuracy(model, eval_dataset, device),
        "size_MB": get_model_size_in_mb(model),
        "inference_time_ms": measure_inference_time(model, tokenizer, eval_dataset, device),
        "gpu_memory": measure_gpu_memory(model, tokenizer, sample_text, device),
        "cpu_memory": measure_cpu_memory(model, tokenizer, sample_text, device),
    }

def save_metrics_csv(metrics_dict, model_key):
    filename=f"{model_key}_compression_metrics.csv"
    reports_dir = Path("compression") / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics_dict)
    df.to_csv(reports_dir / filename, index=False)