from torch.nn.utils import prune
import torch.nn as nn
import copy
import wandb

#linear unstructured pruning
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
        return model

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


# now a pruning function with wandb logging
def global_pruning_linears_with_wandb(
    model, 
    amount: float, 
    make_permanent: bool = True, 
    log_to_wandb: bool = True
):
    """
    Globally prune a fraction `amount` of weights across ALL nn.Linear layers
    (excluding classifier). Optionally logs sparsity stats to wandb.
    """

    # Copy model to avoid modifying original
    model = copy.deepcopy(model)

    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "classifier" not in name:
            linear_layers.append((module, "weight"))

    if not linear_layers:
        print("No eligible Linear layers found for pruning.")
        if log_to_wandb:
            wandb.log({"pruning/amount": amount, "pruning/sparsity_linear": 0.0})
        return model, {"sparsity_linear": 0.0, "total_linear_params": 0}

    # Apply pruning
    prune.global_unstructured(
        linear_layers,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Make pruning permanent
    if make_permanent:
        for layer, _ in linear_layers:
            prune.remove(layer, "weight")

    # Calculate sparsity
    total_params = sum(layer.weight.numel() for layer, _ in linear_layers)
    zero_params = sum((layer.weight == 0).sum().item() for layer, _ in linear_layers)
    sparsity = zero_params / total_params

    # Console log
    print(f"Pruned model sparsity: {sparsity:.2%}")
    print(f"Total pruned parameters: {zero_params} / {total_params}")

    # WandB log
    if log_to_wandb:
        wandb.log({
            "pruning/amount": amount,
            "pruning/sparsity_linear": sparsity,
            "pruning/zero_params": zero_params,
            "pruning/total_linear_params": total_params
        })

    return model, {
        "sparsity_linear": sparsity,
        "zero_params": zero_params,
        "total_linear_params": total_params
    }



# _____________ structured pruning __________________
def _get_encoder_layers(model):
    """Return the list of transformer layers for BERT/RoBERTa."""
    if hasattr(model, "bert"):
        return model.bert.encoder.layer
    if hasattr(model, "roberta"):
        return model.roberta.encoder.layer
    raise ValueError("This helper currently supports BERT/RoBERTa models only.")

def _evenly_spaced_heads(num_heads, keep_ratio):
    """Return the head indices to REMOVE (uniformly spaced)."""
    keep = max(1, int(round(num_heads * keep_ratio)))
    remove = num_heads - keep
    if remove <= 0:
        return []
    # spread removals (e.g., for 12 heads keep_ratio=0.5 -> remove [1,3,5,7,9,11])
    step = num_heads / remove
    idxs = sorted({int(round((i+1)*step - 1)) for i in range(remove)})
    # clamp to [0, num_heads-1]
    return [min(max(i, 0), num_heads - 1) for i in idxs]

# ---- main pruning function ----
def prune_attention_heads_with_wandb(
    model,
    keep_ratio_per_layer: float = 0.5,
    log_to_wandb: bool = True,
):
    """
    Structured head pruning for BERT/RoBERTa (no fine-tuning required).
    Keeps `keep_ratio_per_layer` of heads in every layer (uniformly spaced),
    removes the rest using HF's built-in `prune_heads`.

    Returns: (pruned_model, stats_dict)
    """
    assert 0.0 < keep_ratio_per_layer <= 1.0, "keep_ratio_per_layer must be in (0,1]."

    pruned = copy.deepcopy(model).eval()
    layers = _get_encoder_layers(pruned)
    num_layers = pruned.config.num_hidden_layers
    num_heads = pruned.config.num_attention_heads
    head_dim = pruned.config.hidden_size // num_heads

    total_heads = num_layers * num_heads
    total_remove = 0
    per_layer_removed = {}

    # build per-layer removal plan
    to_prune = {}
    for l in range(num_layers):
        remove_idxs = _evenly_spaced_heads(num_heads, keep_ratio_per_layer)
        if len(remove_idxs) > 0:
            to_prune[l] = sorted(set(remove_idxs))
            per_layer_removed[l] = len(to_prune[l])
            total_remove += len(to_prune[l])

    # try layer-level pruning first (BertAttention has prune_heads)
    layer_level_ok = all(
        hasattr(layers[l].attention, "prune_heads") for l in to_prune.keys()
    )

    if layer_level_ok:
        for l, heads in to_prune.items():
            # IMPORTANT: call on attention (not .self) for Sdpa variants
            layers[l].attention.prune_heads(set(heads))
    elif hasattr(pruned, "prune_heads"):
        # Fallback: some models expose model.prune_heads({layer: set(heads)})
        pruned.prune_heads({l: set(h) for l, h in to_prune.items()})
    else:
        raise AttributeError(
            "This model does not expose a head-pruning method on layers or model."
        )

    total_heads = num_layers * num_heads
    kept_heads = total_heads - total_remove

    # rough parameter accounting (Q,K,V,O projections):
    # each head carries 4 * (hidden_size * head_dim) params per layer (q,k,v,o)
    # but prune_heads rewires inside the projections; report head counts instead
    stats = {
        "pruning/heads_total": total_heads,
        "pruning/heads_removed_total": total_remove,
        "pruning/heads_kept_total": kept_heads,
        "pruning/keep_ratio_per_layer": keep_ratio_per_layer,
        "model/num_layers": num_layers,
        "model/num_heads_per_layer": num_heads,
        "model/head_dim": head_dim,
    }

    # log simple scalars
    if log_to_wandb:
        wandb.log(stats)

        # optional: per-layer removal breakdown as individual metrics
        for l, nrem in per_layer_removed.items():
            wandb.log({f"pruning/heads_removed_layer_{l}": nrem})

    print(
        f"[Head Pruning] kept {kept_heads}/{total_heads} heads "
        f"({100*kept_heads/total_heads:.1f}%), "
        f"removed {total_remove} heads across {num_layers} layers."
    )

    return pruned, stats