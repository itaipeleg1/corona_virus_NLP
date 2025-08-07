from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import wandb
import os
from models import data_preparation

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'recall': recall,
        'precision': precision,
    }

def objective_HF(trial, tokenizer, model_name, model_class, base_attr, project_name, training_type, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameter suggestions - WE SHOULD ADD MORE HYPERPARAMETERS HERE
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-4)
    patience = trial.suggest_int("patience", 7, 10)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3, step=1)
    dropout = trial.suggest_float("dropout_rate", 0.1, 0.5)

    #build the model:
    config = AutoConfig.from_pretrained(
    model_name,
    num_labels=5,
    hidden_dropout_prob=dropout,
    attention_probs_dropout_prob=dropout
    )

    model = model_class.from_pretrained(model_name, config=config).to(device)

    # Freeze layers if base_attr is valid
    base_model = getattr(model, base_attr, None)
    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.encoder.layer[-num_layers:].parameters():
        param.requires_grad = True

    # Prepare datasets
    train_dataset, eval_dataset, _ = data_preparation.prepare_dataset(tokenizer, max_length)

    #CHANGE AND ADD STRATIFIED K FOLD###


    # Configure W&B
    wandb.init(project=project_name,
               config={ 
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "architecture": model_name,
        "training_type": training_type,
        "dataset": "corona_virus_NLP"}, 
        name=f"{project_name}_{training_type}_trial_{trial.number}") # The name that will be saved in the W&B platform

    # Hugging Face TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"./results/{project_name}_{training_type}_trial_{trial.number}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=20,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=["wandb"],
        logging_steps=10,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    )

    trainer.train()
    metrics = trainer.evaluate()
    wandb.finish()

    return metrics["eval_accuracy"]