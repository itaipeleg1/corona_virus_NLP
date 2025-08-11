from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForSequenceClassification, AutoConfig, RobertaForSequenceClassification, BertForSequenceClassification, RobertaTokenizer, BertTokenizer
from transformers.models.bertweet import BertweetTokenizer



model_configs = {
    "bertweet": {
        "model_name": "vinai/bertweet-base",
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": BertweetTokenizer,
        "base_attr": "roberta",
        "max_length": 128  # critical for bertweet
    },
    "covidbert": {
        "model_name": "digitalepidemiologylab/covid-twitter-bert-v2",
        "model_class": BertForSequenceClassification,
        "tokenizer_class": BertTokenizer,
        "base_attr": "bert",
        "max_length": 128  ##critical for covidbert
    },
    "roberta": {
        "model_name": "roberta-base",
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": RobertaTokenizer,
        "base_attr": "roberta",
        "max_length": 512 # critical for roberta
    },
}
