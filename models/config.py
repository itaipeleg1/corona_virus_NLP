from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForSequenceClassification, AutoConfig, RobertaForSequenceClassification, BertForSequenceClassification, RobertaTokenizer, BertTokenizer

#LATER WE SHOULD IMPLEMENT THE MODELS USING THESE DICTIONARRIES


model_configs = {
    "bertweet": {
        "model_name": "vinai/bertweet-base",
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": RobertaTokenizer,
        "base_attr": "roberta"
    },
    "covidbert": {
        "model_name": "digitalepidemiologylab/covid-twitter-bert",
        "model_class": BertForSequenceClassification,
        "tokenizer_class": BertTokenizer,
        "base_attr": "bert"
    }
}

training_type = ["pytorch", "hf_trainer"]

study_name = f"{model_name}_{training_type}_study"

#more things: which metric we use for optuna? and which direction?