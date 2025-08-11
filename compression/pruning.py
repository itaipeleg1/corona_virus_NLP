from torch.nn.utils import prune

def global_pruning_linears(model, amount: float, make_permanent=True):
    """
    Globally prune a fraction `amount` of weights across ALL nn.Linear layers.
    
    """
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
    return model, {"sparsity_linear": sparsity, "total_linear_params": total_params}


    # #saving pruned_model  - for later:
    # os.makedirs(output_dir, exist_ok=True)
    # save_path = os.path.join(output_dir, filename)
    # torch.save(model.state_dict(), save_path)
    # print(f"✅ Pruned model saved to: {save_path}")

   
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

