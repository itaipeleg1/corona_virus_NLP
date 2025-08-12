from torch.nn.utils import prune
import torch.nn as nn
import copy
def global_pruning_linears(model, amount: float, make_permanent=True):
    """
    Globally prune a fraction `amount` of weights across ALL nn.Linear layers.
    
    """
    #copies model to avoid pruning original model
    model = copy.deepcopy(model)
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "classifier" not in name: # not pruning classififer because of accuracy drop
            linear_layers.append((module, "weight"))

    if not linear_layers:
        print("No eligible Linear layers found for pruning.")
        return model, {"sparsity_linear": 0.0, "total_linear_params": 0}

    prune.global_unstructured(
        linear_layers,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    if make_permanent:
        for layer, _ in linear_layers:
            prune.remove(layer, "weight")

    total_params = sum(layer.weight.numel() for layer, _ in linear_layers)
    zero_params = sum((layer.weight == 0).sum().item() for layer, _ in linear_layers)
    sparsity = zero_params / total_params

    print(f"Pruned model sparsity: {sparsity:.2%}")
    print(f"Total pruned parameters: {zero_params} out of {total_params} ({sparsity:.2%})")
    return model
