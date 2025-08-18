from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForSequenceClassification, AutoConfig, RobertaForSequenceClassification, BertForSequenceClassification, RobertaTokenizer, BertTokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
from transformers.models.bertweet import BertweetTokenizer
from config import PROJECT_ROOT


model_configs = {
    "bertweet": {
        "model_name": "vinai/bertweet-base",
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": BertweetTokenizer,
        "base_attr": "roberta",
        "max_length": 128,  # critical for bertweet
        "is_state_dict": True,
        "best_path": PROJECT_ROOT / "results/best_models/bertweet_HF_study_augmented_state_dict.pt",
        "description": "Best Fine-tuned BERTweet mode using HF"
    },
    "covidbert": {
        "model_name": "digitalepidemiologylab/covid-twitter-bert",
        "model_class": BertForSequenceClassification,
        "tokenizer_class": BertTokenizer,
        "base_attr": "bert",
        "max_length": 128,  ##critical for covidbert
        "is_state_dict": True,
        "best_path": PROJECT_ROOT / "results/best_models/covidbert_pytorch_study_augmented_state_dict.pt",
        "description": "Full fine-tuned CovidBERT model"
    },
        "Distilled_BerTweet": { # change in main_compress and ui from Compressed BertWeet
        "model_name": "distilroberta-base",
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": RobertaTokenizer,
        "base_attr": "roberta",
        "max_length": 128,
        "is_state_dict": True,
        "best_path": PROJECT_ROOT / "results/best_compressed/bertweet_knowledge_distillation_model.pt",
        "description": "Distilled BERTweet model"
    },

    "Distilled_CovidBert": {
        "model_name": "distilbert-base-uncased",
        "model_class": DistilBertForSequenceClassification,
        "tokenizer_class": DistilBertTokenizer,
        "base_attr": "bert",
        "max_length": 128,
        "model_name": "distilbert-base-uncased",
        "is_state_dict": True,
        "best_path": PROJECT_ROOT / "results/best_compressed/covidbert_knowledge_distillation_model.pt",
        "description": "Distilled CovidBERT model"
    }
}
