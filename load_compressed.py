#loading all covidbert_HF models
import torch
import copy
import pandas as pd
from compression.model_loading import load_pt_model, load_student_model
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
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForSequenceClassification, AutoConfig, RobertaForSequenceClassification, BertForSequenceClassification, RobertaTokenizer, BertTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
from transformers.models.bertweet import BertweetTokenizer
from config import PROJECT_ROOT
from torch.ao.quantization import quantize_dynamic

#model_key = "covidbert_HF"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_quantized_model(model_key: str, device: str = 'cpu'):
    # Step 1: Create original model
    base_model_name = model_key.split('_')[0]
    config = model_configs[base_model_name]
    model = config["model_class"].from_pretrained(config["model_name"], num_labels=5)
    
    # Step 2: Apply quantization (same as during training)
    model.eval()
    quantized_model = quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Step 3: Load the quantized state_dict
    state_dict_path = COMPRESSION_OUTPUT_DIR / model_key / f'{model_key}_quantization_state_dict.pt'
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=False)
    quantized_model.load_state_dict(state_dict)
    
    return quantized_model

# Usage:
q_model = load_quantized_model('covidbert_HF', device='cpu')
print("Quantized model loaded successfully!")


# def load_distilled_model(model_key: str = "covidbert_HF", device: str = 'cpu'):
#     """Load distilled model - uses student architecture (DistilBERT)"""
    
#     # Get student config from the base model config
#     base_model_name = model_key.split('_')[0]
#     student_key = model_configs[base_model_name]["student_key"]  # "distilbert"
#     student_config = model_configs[student_key]
    
#     # Create student model architecture
#     model = student_config["model_class"].from_pretrained(student_config["model_name"], num_labels=5)
#     tokenizer = student_config["tokenizer_class"].from_pretrained(student_config["model_name"])
    
#     # Load distilled weights
#     distilled_path = compression_configs[model_key]["knowledge_distillation_path"]
#     state_dict = torch.load(distilled_path, map_location=device, weights_only=False)
    
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
    
#     print(f"✅ Distilled model loaded from {distilled_path}")
#     return model, tokenizer

# # Test both
# try:
#     print("\n=== Testing Distilled Model ===")
#     distilled_model, distilled_tokenizer = load_distilled_model("covidbert_HF", device='cpu')
#     print(f"Distilled model size: {sum(p.numel() for p in distilled_model.parameters())} parameters")
    
#     print("\n✅ Both models loaded successfully!")
    
# except Exception as e:
#     print(f"❌ Error loading models: {e}")
#     import traceback
#     traceback.print_exc()

# def load_pruned_model(model_key: str = "covidbert_HF", device: str = 'cpu'):
#     """Load full pruned model since pruning changed the architecture"""
    
#     # Load the full pruned model (not state_dict)
#     pruned_full_path = COMPRESSION_OUTPUT_DIR / model_key / f'{model_key}_pruning_full.pt'
    
#     try:
#         # First try with safe globals
#         from transformers.models.bert.modeling_bert import BertForSequenceClassification
#         with torch.serialization.safe_globals([BertForSequenceClassification]):
#             model = torch.load(pruned_full_path, map_location=device)
#     except:
#         # Fallback to weights_only=False
#         model = torch.load(pruned_full_path, map_location=device, weights_only=False)
    
#     # Load tokenizer separately (pruning doesn't affect tokenizer)
#     base_model_name = model_key.split('_')[0] 
#     config = model_configs[base_model_name]
#     tokenizer = config["tokenizer_class"].from_pretrained(config["model_name"])
    
#     model.to(device)
#     model.eval()
    
#     print(f"✅ Pruned model loaded from {pruned_full_path}")
#     return model, tokenizer


# try:
#     print("=== Testing Pruned Model ===")
#     pruned_model, pruned_tokenizer = load_pruned_model("covidbert_HF", device='cpu')
#    print(f"Pruned model size: {sum(p.numel() for p in pruned_model.parameters())} parameters")
    
# except Exception as e:
#     print(f"❌ Error loading models: {e}")
#     import traceback
#     traceback.print_exc()

#load only quantized model:
# full_model_path =  COMPRESSION_OUTPUT_DIR / 'covidbert_HF' / 'covidbert_HF_quantization_full.pt'


# state_dict_path = COMPRESSION_OUTPUT_DIR / 'covidbert_HF' / 'covidbert_HF_quantization_state_dict.pt'

# try:
#     state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=False)
#     print("State dict loaded successfully")
#     print(f"Number of parameters: {len(state_dict)}")
#     # Check if it contains quantized parameters
#     quantized_keys = [k for k in state_dict.keys() if any(x in k for x in ['scale', 'zero_point', '_packed_params'])]
#     print(f"Quantized parameter keys found: {len(quantized_keys)}")
# except Exception as e:
#     print(f"State dict also corrupted: {e}")
#q_model = torch.load(full_model_path, map_location=device, weights_only=False)
#print('model')



# original_dict_path = compression_configs[model_key]["base_path"] #access saved best fine tuned model path
    
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")



# # Load model configuration
# base_model_name = model_key.split('_')[0] #turn "bertweet_HF" to "bertweet" to access model architecture from model_configs
# print(f"Base model name: {base_model_name}")

# config = model_configs[base_model_name]
# max_length = config["max_length"]
# student_key = config["student_key"]
# print(f"student key: {student_key}")
# # checkpoint = torch.load(model_dict_path, map_location='cpu')
# # print(f'checkpoint keys: {checkpoint.keys()}')

# # Load original model
# print("\n=== LOADING ORIGINAL MODEL ===")
# original_model, original_tokenizer = load_pt_model(
#     model_key=model_key, 
#     dict_path=original_dict_path, 
#     num_labels=5, 
#     device=device
# )

#     #load pretrained models using state dict paths

# print("\n=== LOADING COMPRESSED MODELS ===")
# compressed_dict = load_compressed_models(model_key=model_key, student_key=student_key, num_labels=5, device=device)

# # Unpack models and tokenizers
# q_model, _ = compressed_dict["quantized"]
# p_model, _ = compressed_dict["pruned"]
# distilled_model, student_tokenizer = compressed_dict["distilled"]

# print ("loading was succesful")

