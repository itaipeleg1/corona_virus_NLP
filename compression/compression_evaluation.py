# comparisons: model_size, inference_time, memory usage and accuracy
import torch
import time
import os
import psutil
import numpy as np
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

def evaluate_performance(model, test_dataset, device, n_classes=5, batch_size=32, max_samples=60):

    print(f"Starting accuracy evaluation on {device}")
    print(f"Dataset length: {len(test_dataset)}")
    model.to(device)
    model.eval()
    preds = []
    labels = []
    probs = []
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    seen = 0
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].cpu().numpy()

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits  # (1, n_classes)
            probs_tensor = torch.softmax(logits, dim=-1).cpu().numpy()  # (1, n_classes)
            batch_preds = np.argmax(probs_tensor, axis=-1)  # (1,)

        preds.extend(batch_preds)
        labels.extend(batch_labels)
        probs.append(probs_tensor)  # move to CPU and convert to numpy
        
        seen += input_ids.size(0)
        if max_samples is not None and seen >= max_samples: 
            break

    all_labels = np.array(labels)[:max_samples]
    all_preds = np.array(preds)[:max_samples]
    all_probs = np.vstack(probs)[:max_samples]

    accuracy = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    #now computing auc:
    # AUC for multiclass
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted') # ovr - onr vs rest - computes AUC for each class and the a weighted average
    
    except:
        auc = None
        print("Warning: Could not calculate AUC")
    
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
    print(cm)                        
    print(f"Accuracy evaluation complete: {len(preds)} predictions made")
    print(f'accuracy: {accuracy:.4f}, f1_macro: {f1_macro:.4f}, f1_weighted: {f1_weighted:.4f}, auc: {auc:.4f}')

    return {'accuracy': accuracy,
            'f1 macro': f1_macro,
            'f1 weighted': f1_weighted,
            'auc': auc,
            'confusion_matrix': cm,
    }

def get_model_size_in_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / 1024**2
    return total_size_mb

def measure_inference_time(model, test_dataset, device, runs=100, batch_size=8):
    model.to(device)
    model.eval()

    # Warmup - avoiding inital overheasd by warming up and not measuring
    for _ in range(5):
        batch = [test_dataset[i] for i in range(batch_size)]
        input_ids = torch.stack([b['input_ids'] for b in batch]).to(device)
        attention_mask = torch.stack([b['attention_mask'] for b in batch]).to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

    # now evaluate inference time
    start_time = time.time()
    for _ in range(runs):
        batch = [test_dataset[i] for i in range(batch_size)]
        input_ids = torch.stack([b['input_ids'] for b in batch]).to(device)
        attention_mask = torch.stack([b['attention_mask'] for b in batch]).to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    total_time = time.time() - start_time
    avg_time_ms = (total_time / runs) * 1000
    return round(avg_time_ms, 2)


# def measure_gpu_memory(model, tokenizer, sample_text, device='cuda'):
#     if device != 'cuda' or not torch.cuda.is_available():
#         print("GPU not available. Skipping GPU memory measurement.")
#         return None

#     torch.cuda.reset_peak_memory_stats(device)
#     model.to(device)
#     model.eval()
#     inputs = tokenizer(sample_text, return_tensors="pt").to(device)

#     with torch.no_grad():
#         _ = model(**inputs)

#     mem_used_mb = torch.cuda.max_memory_allocated(device) / 1024**2
#     return round(mem_used_mb, 2)

# # def measure_cpu_memory(model, tokenizer, sample_text, device='cpu'):
# #     model.to(device)
# #     model.eval()
# #     inputs = tokenizer(sample_text, return_tensors="pt").to(device)

# #     process = psutil.Process(os.getpid())
# #     mem_before = process.memory_info().rss
# #     with torch.no_grad():
# #         _ = model(**inputs)
# #     mem_after = process.memory_info().rss
# #     mem_used_mb = (mem_after - mem_before) / (1024 * 1024)
# #     return round(mem_used_mb, 2)


# def measure_cpu_memory(model, test_dataset, device="cpu", batch_size=32, max_samples=60, include_warmup=True):
#     """
#     Measures peak process RSS (MB) during inference over up to `max_samples` examples.
#     Assumes each dataset item is a dict of tensors with keys like input_ids, attention_mask, (labels optional).
#     """
#     model.to("cpu").eval()
#     loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # default collate stacks dicts of tensors
#     proc = psutil.Process(os.getpid()) #gets my currect progress for memory usage

#     # warmup phase
#     if include_warmup:
#         with torch.inference_mode():
#             for batch in loader:
#                 inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
#                 _ = model(**inputs)
#                 break  

#     # baseline after warmup
#     baseline = proc.memory_info().rss #gives resident set size (RSS) in bytes (physical memory)
#     peak = baseline #maximum memory used
#     seen = 0 #  amount of seen samples so far

#     with torch.inference_mode():
#         for batch in loader:
#             inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
#             _ = model(**inputs)
#             current_memory = proc.memory_info().rss #measures
#             if current_memory > peak:
#                 peak = current_memory

#             seen += next(iter(inputs.values())).size(0)  # batch size from any tensor
#             if max_samples is not None and seen >= max_samples:
#                 break

#     return round((peak - baseline) / (1024**2), 3) #MB

def evaluate_model(model, tokenizer, test_dataset, sample_text, device: str):

    metrics = evaluate_performance(model, test_dataset, device)
    
    return {
        "accuracy": metrics['accuracy'],
        "auc": metrics['auc'],
        "f1_macro": metrics['f1 macro'],
        "f1_weighted": metrics['f1 weighted'],
        "confusion_matrix": metrics.get('confusion_matrix', None),
        "size_MB": get_model_size_in_mb(model),
        "inference_time_ms": measure_inference_time(model, test_dataset, device),
        # "gpu_memory": measure_gpu_memory(model, tokenizer, sample_text, device),
        # "cpu_memory": measure_cpu_memory(model, test_dataset, device),
    }

def save_metrics_csv(metrics_dict, model_key):
    filename=f"{model_key}_compression_metrics.csv"
    reports_dir = Path("compression") / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics_dict).T #rows=models, cols=metrics
    df.index.name = "model"
    df.reset_index().to_csv(reports_dir / filename, index=False)