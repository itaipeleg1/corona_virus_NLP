import torch
import copy
import pandas as pd
from compression.model_loading import load_pt_model, load_student_model, load_distilled_model
from compression.quantization import quantize_model
from compression.pruning import prune_attention_heads_with_wandb
from compression.distillation_HF import knowledge_distillation
from compression.compression_evaluation import evaluate_model, save_metrics_csv
from compression.save_compressed_models import save_model_state
#from compression.compression_config import COMPRESSION_OUTPUT_DIR
from config import DATA_DIR, COMPRESSION_OUTPUT_DIR
from models.model_config import model_configs
from models.data_preparation import prepare_dataset
from compression.compression_configs import compression_configs

def main(model_key, distill_epochs: int, do_train: bool, do_save_models: bool, do_save_reports: bool, temperature: float, alpha: float, max_samples: int = None,):
    print(f"Starting compression pipeline for {model_key}")
    print("="*60)
    
    original_dict_path = compression_configs[model_key]["base_path"] #access saved best fine tuned model path
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model configuration
    base_model_name = model_key.split('_')[0] #turn "bertweet_HF" to "bertweet" to access model architecture from model_configs
    print(f"Base model name: {base_model_name}")

    config = model_configs[base_model_name]
    max_length = config["max_length"]
    student_key = config["student_key"]

    # Load student model (before compressions)
    print("\n=== LOADING STUDENT MODEL - before distillation ===")
    student_model, student_tokenizer = load_student_model(student_key=student_key, num_labels=5, device=device)

    # Load original model
    print("\n=== LOADING ORIGINAL MODEL ===")
    original_model, original_tokenizer = load_pt_model(
        model_key=model_key, 
        dict_path=original_dict_path, 
        num_labels=5, 
        device=device
    )

    # Apply compressions (NOW we have everything loaded)
    print("\n=== APPLYING COMPRESSIONS ===")
    
    # QUANTIZATION
    print("\n--- Quantization ---")
    q_model = quantize_model(copy.deepcopy(original_model), model_key,  dtype=torch.qint8)

    # PRUNING
    print("\n--- Pruning ---")
    p_model, _ = prune_attention_heads_with_wandb(
    model=original_model,
    keep_ratio_per_layer=0.75, #ratio of head we keep after hyperparameter tuning and logging in wandb
    log_to_wandb=False
    )
    #old version of global pruning - maybe switch back or make modular
    #p_model = global_pruning_linears(copy.deepcopy(original_model), amount=amount, make_permanent=True)
    
    if not do_train:
        distilled_model, _ = load_distilled_model(model_key=model_key, device=device)

    else:
        
        # KNOWLEDGE DISTILLATION
        print("\n--- Knowledge Distillation ---")
        distilled_model, _ = knowledge_distillation(
            student_key, model_key, student_model, student_tokenizer, original_model,
            temperature=temperature, alpha=alpha, epochs=distill_epochs, output_dir=COMPRESSION_OUTPUT_DIR
        )
        if do_save_models:
            save_model_state(distilled_model, model_key, compression_type="knowledge_distillation", output_dir=COMPRESSION_OUTPUT_DIR)


    # COMPREHENSIVE EVALUATION 
    print("\n=== EVALUATION PHASE ===")
    
        # Prepare datasets for evaluation (NOW we have both tokenizers)
    print("\n=== PREPARING DATASETS ===")
    _, _, test_dataset = prepare_dataset(original_tokenizer, max_length=max_length)
    _, _, student_test_dataset = prepare_dataset(student_tokenizer, max_length=max_length)

    print("\n--- Evaluating Original Model ---")
    original_metrics = evaluate_model(original_model, test_dataset, device=device, max_samples=max_samples)
    
    print("\n--- Evaluating Quantized Model ---")
    quantized_metrics = evaluate_model(q_model,  test_dataset, device='cpu', max_samples=max_samples)  # Quantized models must use CPU

    print("\n--- Evaluating Pruned Model ---")
    pruned_metrics = evaluate_model(p_model, test_dataset,  device=device, max_samples=max_samples)

    print("\n--- Evaluating Distilled Model ---")
    distilled_metrics = evaluate_model(distilled_model, student_test_dataset, device=device, max_samples=max_samples)

    metrics_dict = {
        "original": original_metrics,
        "quantized": quantized_metrics,
        "pruned": pruned_metrics,
        "distilled": distilled_metrics
    }
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("COMPRESSION RESULTS SUMMARY")
    print("="*60)
    
    for method, metrics in metrics_dict.items():
        print(f"\n{method.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 macro: {metrics['f1_macro']:.4f}")
        print(f"  F1 weighted: {metrics['f1_weighted']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
        print(f"  Size (MB): {metrics['size_MB']:.2f}")
        print(f"  Inference Time (ms) on cpu: {metrics['inference_time_ms_cpu']}")
        print(f"  Inference Time (ms) on GPU: {metrics['inference_time_ms_gpu']}")

    
    # Calculate compression ratios
    print(f"\n" + "-"*40)
    print("COMPRESSION RATIOS (vs Original):")
    print("-"*40)
    orig_size = original_metrics['size_MB']
    orig_time = original_metrics['inference_time_ms_cpu']
    
    for method, metrics in metrics_dict.items():
        if method != 'original':
            size_ratio = metrics['size_MB'] / orig_size
            time_ratio = metrics['inference_time_ms_cpu'] / orig_time if orig_time > 0 else 1
            acc_drop = original_metrics['accuracy'] - metrics['accuracy']
            
            print(f"{method}: Size={size_ratio:.2f}x, Speed={time_ratio:.2f}x, Acc_drop={acc_drop:.3f}")
    
    if do_save_reports:
        save_metrics_csv(metrics_dict, model_key)
        print(f"\nMetrics saved to compression/reports/{model_key}_compression_metrics.csv")
    
    return metrics_dict

if __name__ == "__main__":
    # Configuration
    # model_keys = ["bertweet", "covidbert"]
    model_keys = ["bertweet_HF", "bertweet_pytorch", "covidbert_HF", "bertweet_pytorch"]
    compression_types = ['quantization', 'pruning', 'knowledge_distillation']
    
    # Compression hyperparameters 
    temperature = 3.0  # distillation temperature
    alpha = 0.7  # distillation alpha
    
    # Current model to process (change this to switch between models)

    
    # Training parameters
    distill_epochs = 5 #to change to 5
    
    try:
        for model_key in compression_configs.keys(): 
            results = main(
                model_key=model_key, 
                distill_epochs=distill_epochs,
                do_train=False, #if not training - loading state dicts from the already compressed models 
                do_save_models=False, 
                do_save_reports=False, # reports can be viewed in "reports" folder
                temperature=temperature,
                alpha=alpha,
                max_samples=30
            )
                # Run compression pipeline
    
    
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            
    except Exception as e:
        print(f"\nERROR: Pipeline failed with: {e}")
        import traceback
        traceback.print_exc()