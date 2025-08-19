import wandb
from compression.pruning import global_pruning_linears_with_wandb, prune_attention_heads_with_wandb
from compression.model_loading import load_pt_model
from compression.compression_configs import compression_configs
from compression.compression_evaluation import evaluate_performance
from models.data_preparation import prepare_dataset
import torch

amounts = [0.1, 0.3, 0.5, 0.7]
device = "cuda" if torch.cuda.is_available() else "cpu"
model_key = "bertweet_pytorch"
original_dict_path = compression_configs[model_key]["base_path"]
original_model, original_tokenizer = load_pt_model(model_key=model_key, dict_path=original_dict_path, num_labels=5, device=None)
_, _, test_dataset = prepare_dataset(original_tokenizer, max_length=128)

ratios = [0.33, 0.5, 0.75]
for ratio in ratios:
    wandb.init(project="structured_pruning_experiments", name=f"{model_key}_heads_keep_ratio_{ratio}")
    pruned_model, head_stats = prune_attention_heads_with_wandb(
        model=original_model,
        keep_ratio_per_layer=ratio, #ratio is how many heads we keep
        log_to_wandb=True
    )
    metric_dict = evaluate_performance(model=pruned_model, test_dataset=test_dataset, device=device)
    wandb.log({"accuracy": metric_dict["accuracy"],
               "f1 weighted": metric_dict["f1 weighted"]})
    wandb.finish()

# for amt in [0.1, 0.3, 0.5, 0.7]:
#     wandb.init(project="pruning_experiments", name=f"{model_key}_prune_{amt}")
#     pruned_model, stats = global_pruning_linears_with_wandb(
#         model=original_model, amount=amt, make_permanent=True, log_to_wandb=True
#     )
#     metric_dict = evaluate_performance(model=pruned_model, test_dataset=test_dataset, device=device)
#     wandb.log({"accuracy": metric_dict["accuracy"],
#                "f1 weighted": metric_dict["f1 weighted"]})
#     wandb.finish()