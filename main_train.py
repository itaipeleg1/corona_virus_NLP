from transformers import RobertaForSequenceClassification, BertForSequenceClassification
from transformers import RobertaTokenizer, BertTokenizer
from models.training_pytorch import objective
from models.training_HF import objective_HF
from models.model_config import model_configs
import optuna
import wandb
import argparse
from dotenv import load_dotenv
import os

def main(study_name: str, model_key: str, training_type: str):
    # Load config
    config = model_configs[model_key]
    model_name = config["model_name"]
    model_class = config["model_class"]
    tokenizer_class = config["tokenizer_class"]
    base_attr = config["base_attr"]

    ##Log in to Weights & Biases
    load_dotenv()  # Load environment variables from a .env file
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # Initialize model and tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name)    
 

    # Create Optuna study
    study = optuna.create_study(study_name=study_name, direction="maximize")

    # Optimize
    if training_type == "pytorch":
        study.optimize(
            lambda trial: objective(
                trial=trial,
                tokenizer=tokenizer,
                model_name=model_name,
                model_class=model_class,
                base_attr=base_attr,
                project_name=study_name,
                training_type=training_type,
                max_length=config["max_length"]
            ),
            n_trials=3
        )
    elif training_type == "HF":
        study.optimize(
            lambda trial: objective_HF(
                trial=trial,
                tokenizer=tokenizer,
                model_name=model_name,
                model_class=model_class,
                base_attr=base_attr,
                project_name=study_name,
                training_type=training_type,
                max_length=config["max_length"]
            ),
            n_trials=3
        )
    else:
        raise ValueError(f"Unsupported training type: {training_type}")
    
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number} with value: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str,  help="Name of the Optuna study / W&B project")
    parser.add_argument("--model_key", type=str, default="bertweet", choices=model_configs.keys(), help="Which model to use")
    parser.add_argument("--training_type", type=str, default="pytorch", help="Training type tag (pytorch / HF)")

    args = parser.parse_args()
    model_keys = [ "covidbert", "bertweet"]
    training_types = ["HF", "pytorch"]

    for model_key in model_keys:
        for training_type in training_types:
            print(f"Running study for model: {model_key}, training type: {training_type}")
            args.study_name = f"run_2.3_{model_key}_{training_type}"
            args.model_key = model_key
            args.training_type = training_type
            # Call the main function with the current model key and training type
            main(args.study_name, args.model_key, args.training_type)