from config import PROJECT_ROOT, COMPRESSION_OUTPUT_DIR

# here i will add all best dicts from compression / saved_compressed
compression_configs = {
    "covidbert_HF": {
        "base_path": PROJECT_ROOT / "results/best_models/covidbert_HF_study_augmented_state_dict.pt",
        "quantization_path": COMPRESSION_OUTPUT_DIR / 'covidbert_HF' / 'covidbert_HF_quantization_model.pt',
        "pruning_path": COMPRESSION_OUTPUT_DIR / 'covidbert_HF' / 'covidbert_HF_pruning_model.pt',
        "knowledge_distillation_path": COMPRESSION_OUTPUT_DIR / 'covidbert_HF' / 'covidbert_HF_knowledge_distillation_model.pt',
        "distillation_summary" : COMPRESSION_OUTPUT_DIR / 'covidbert_HF' / 'covidbert_HF_knowledge_distillation_summary.json'


    },
    "covidbert_pytorch": {
        "base_path": PROJECT_ROOT / "results/best_models/covidbert_pytorch_study_augmented_state_dict.pt",
        "quantization_path": COMPRESSION_OUTPUT_DIR / 'covidbert_pytorch' / 'covidbert_pytorch_quantization_model.pt',
        "pruning_path": COMPRESSION_OUTPUT_DIR / 'covidbert_pytorch' / 'covidbert_pytorch_pruning_model.pt',
        "knowledge_distillation_path": COMPRESSION_OUTPUT_DIR / 'covidbert_pytorch' / 'covidbert_pytorch_knowledge_distillation_model.pt',
        "distillation_summary" : COMPRESSION_OUTPUT_DIR / 'covidbert_pytorch' / 'covidbert_pytorch_knowledge_distillation_summary.json'


    },
    "bertweet_HF": {
        "base_path": PROJECT_ROOT / "results/best_models/bertweet_HF_study_augmented_state_dict.pt",
        "quantization_path": COMPRESSION_OUTPUT_DIR / 'bertweet_HF' / 'bertweet_HF_quantization_model.pt',
        "pruning_path": COMPRESSION_OUTPUT_DIR / 'bertweet_HF' / 'bertweet_HF_pruning_model.pt',
        "knowledge_distillation_path": COMPRESSION_OUTPUT_DIR / 'bertweet_HF' / 'bertweet_HF_knowledge_distillation_model.pt',
        "distillation_summary" : COMPRESSION_OUTPUT_DIR / 'bertweet_HF' / 'bertweet_HF_knowledge_distillation_summary.json'


    },
    "bertweet_pytorch": {
        "base_path": PROJECT_ROOT / "results/best_models/bertweet_pytorch_study_augmented_state_dict.pt",
        "quantization_path": COMPRESSION_OUTPUT_DIR / 'bertweet_pytorch' / 'bertweet_pytorch_quantization_model.pt',
        "pruning_path": COMPRESSION_OUTPUT_DIR / 'bertweet_pytorch' / 'bertweet_pytorch_pruning_model.pt',
        "knowledge_distillation_path": COMPRESSION_OUTPUT_DIR / 'bertweet_pytorch' / 'bertweet_pytorch_knowledge_distillation_model.pt',
        "distillation_summary" : COMPRESSION_OUTPUT_DIR / 'bertweet_pytorch' / 'bertweet_pytorch_knowledge_distillation_summary.json'
    }
}
