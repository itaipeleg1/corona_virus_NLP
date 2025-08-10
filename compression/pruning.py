from torch.nn.utils import prune

def global_pruning_linears(model, amount=0.2, make_permanent=True):
    """
    Globally prune a fraction `amount` of weights across ALL nn.Linear layers.
    Returns (model, {"sparsity_linear": ..., "total_linear_params": ...})
    """
    # pick Linear weights
    to_prune = [(m, "weight") for m in model.modules() if isinstance(m, nn.Linear)]
    if not to_prune:
        print("No nn.Linear layers found. Nothing to prune.")
        return model, {"sparsity_linear": 0.0, "total_linear_params": 0}

    # apply global unstructured pruning
    prune.global_unstructured(
        to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # optionally bake masks into real zeros
    if make_permanent:
        for m, _ in to_prune:
            prune.remove(m, "weight")

    # report sparsity over Linear weights only
    zeros = sum((m.weight == 0).sum().item() for m, _ in to_prune)
    total = sum(m.weight.numel() for m, _ in to_prune)
    print(f"Sparsity in Linear weights: {zeros/total:.2%}  ({zeros}/{total} zeros)")

    return model, {"sparsity_linear": zeros / total, "total_linear_params": total}

# def global_pruning_linears(model, amount=0.2):
#     print("Original model size:", sum(p.numel() for p in model.parameters()))
#     p_model = model
#     #pruning linear layers - removing x% of weights
#     parameters_to_prune = [
#         (module, 'weight')
#         for module in p_model.modules() if isinstance(module, nn.Linear)]

#     prune.global_unstructured(
#         parameters_to_prune,
#         pruning_method=prune.L1Unstructured,
#         amount=amount)
#     p_model.remove(module, 'weight')
#     print("Pruned model non-zero parameters:", sum((p != 0).sum().item() for p in p_model.parameters()))
#     return p_model, {
#         "sparsity_linear": sum((p == 0).sum().item() for p in p_model.parameters()) / sum(p.numel() for p in p_model.parameters()),
#         "total_linear_params": sum(p.numel() for p in p_model.parameters())
#     }

#chnage P - save new model
# def global_pruning_linears(model, amount=0.2, keep_classifier=True, make_permanent=True):
#     to_prune = []
#     for name, m in model.named_modules():
#         if isinstance(m, nn.Linear):
#             if keep_classifier and ("classifier" in name or "score" in name):
#                 continue
#             to_prune.append((m, "weight"))

#     P.global_unstructured(to_prune, pruning_method=P.L1Unstructured, amount=amount)

#     if make_permanent:
#         for m, _ in to_prune:
#             P.remove(m, "weight")

#     # report sparsity over pruned linears
#     total = nz = 0
#     for m, _ in to_prune:
#         w = m.weight.detach()
#         total += w.numel()
#         nz += (w != 0).sum().item()
#     sparsity = 1 - nz / total
#     print("Pruned model non-zero parameters:", sum((p != 0).sum().item() for p in model_to_prune.parameters()))
#     return model, {"sparsity_linear": sparsity, "total_linear_params": total}

