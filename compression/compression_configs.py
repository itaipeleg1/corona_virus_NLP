from config import PROJECT_ROOT

# here i will add all best dicts from compression / saved_compressed
compression_configs = {
    "covidbert_HF": {
        "base_path": PROJECT_ROOT / "results/best_models/covidbert_HF_study_augmented_state_dict.pt",
        "quantization_path": PROJECT_ROOT / "compression/saved_compressed/covidbert_HF/covidbert_HF_quantization_model.pt",
        "pruning_path": PROJECT_ROOT / "compression/saved_compressed/covidbert_HF/covidbert_HF_pruning_model.pt",
        "knowledge_distillation_path": PROJECT_ROOT / "compression/saved_compressed/covidbert_HF/covidbert_HF_knowledge_distillation_model.pt"

    },
    "covidbert_pytorch": {
        "base_path": PROJECT_ROOT / "results/best_models/covidbert_pytorch_study_augmented_state_dict.pt"
    },
    "bertweet_HF": {
        "base_path": PROJECT_ROOT / "results/best_models/bertweet_HF_study_augmented_state_dict.pt"
    },
    "bertweet_pytorch": {
        "base_path": PROJECT_ROOT / "results/best_models/bertweet_pytorch_study_augmented_state_dict.pt"
    }
}
