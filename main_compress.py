import torch
import copy

from compression.model_loading import load_pt_model
from compression.quantization import quantize_model
from compression.pruning import global_pruning_linears
#from compression.compression_config import best_covidbert_dict_path, best_bertweet_dict_path 
from models.model_config import model_configs

#these wil be replaced by values from compression_config.py
#placeholder
model_dict_path = "/mnt/hdd/anatkorol/corona_virus_NLP/results/run_2.5_bertweet_pytorch/best_model_trial_1.pt"
model_key = "bertweet" # "covidbert"


def main(model_dict_path, model_key):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    config = model_configs[model_key]
    model_class = config["model_class"]
    tokenizer_class = config["tokenizer_class"]
    model_name = config["model_name"]
    max_length = config["max_length"]

    #maybe make it modular in case there isn't a dict
    checkpoint = torch.load(model_dict_path, map_location=device)
    #if checkpoint is a dictionary, extract all:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        lr_rate = checkpoint["lr_rate"]
        best_acc = checkpoint["best_acc"]
    else:
        state_dict = checkpoint  # assume it's already a state_dict

    # Load original model and relevant hyperparameters
    original_model, tokenizer = load_pt_model(model_key=model_key, model_name=model_name, model_class=model_class, tokenizer_class=tokenizer_class, state_dict=state_dict, num_labels=5, device=device)

    #now perform compressions:
    q_model = quantize_model(original_model, dtype=torch.qint8)

    #pruned model
    p_model, info = global_pruning_linears(original_model, amount=amount, make_permanent=True)

    #adding training to pruned model
    #MISSING!!!

    #knowledge distillation

    #comparison metrics - export to csv



if __name__ == "__main__":
    model_keys = ["bertweet", "covidbert"]
    amount = 0.2 #how much pruning - can be modular
    main(model_dict_path=model_dict_path, model_key=model_key)

#save 3 types of compressions

# compare new models - how?

# def main():
#     # 1) Load base fine-tuned model
#     model, tok = load_pt_model(MODEL_PATH, model_key=MODEL_KEY, num_labels=5)

#     # 2) Quantize (CPU dynamic int8)
#     q_model = quantize_model(model, dtype=torch.qint8)

#     # 3) Prune (unstructured 20% of Linear weights)
#     p_model, p_info = global_pruning_linears(model, amount=0.2, make_permanent=True)

#     # 4) Save compressed models
#     out_root = Path("compressed") / Path(MODEL_PATH).stem
#     pruned_dir = out_root / "pruned_fp32"
#     pruned_dir.mkdir(parents=True, exist_ok=True)
#     p_model.save_pretrained(pruned_dir.as_posix())
#     tok.save_pretrained(pruned_dir.as_posix())

#     # Quantized: export TorchScript artifact (portable CPU)
#     ts_path = out_root / "quantized_int8.ts"
#     max_len = model_configs[MODEL_KEY]["max_length"]
#     example = tok("example text", return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)
#     try:
#         scripted = torch.jit.trace(q_model, (example["input_ids"], example.get("attention_mask"), example.get("token_type_ids")))
#     except Exception:
#         # many HF models accept a dict; fall back to scripting wrapper
#         q_model.eval()
#         scripted = torch.jit.script(q_model)
#     scripted.save(ts_path.as_posix())

#     # 5) Compare (very simple: size + latency)
#     base_bytes = state_bytes(model.state_dict())
#     prun_bytes = state_bytes(p_model.state_dict())
#     # quantized state_dict exists but TorchScript file is the artifact we saved
#     q_latency = quick_latency_ms(q_model, tok, max_length=max_len, n=10)
#     b_latency = quick_latency_ms(model, tok, max_length=max_len, n=10)
#     p_latency = quick_latency_ms(p_model, tok, max_length=max_len, n=10)

#     print("\n=== Comparison ===")
#     print(f"Base size (MB):   {base_bytes/1e6:.2f}")
#     print(f"Pruned size (MB): {prun_bytes/1e6:.2f}  | sparsity_linear={p_info['sparsity_linear']:.2%}")
#     print(f"Quantized TS:     {ts_path}  (load-time dynamic int8)")
#     print(f"Latency avg ms/batch (CPU): base={b_latency:.1f}  pruned={p_latency:.1f}  int8={q_latency:.1f}")

# if __name__ == "__main__":
#     main()