from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
import optuna
import torch
from torch.utils.data import Subset
import numpy as np
import wandb
import os
from models import data_preparation
import traceback
#from models import WeightedLossTrainer

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'recall': recall,
        'precision': precision,
    }


#change the compute_loss method of Trainer from cross entropy to weighted cross entropy
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
    

def objective_HF(trial, tokenizer, model_name, model_class, base_attr, project_name, training_type, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Hyperparameter suggestions - WE SHOULD ADD MORE HYPERPARAMETERS HERE
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-4)
    patience = trial.suggest_int("patience", 7, 10)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3, step=1)
    dropout = trial.suggest_float("dropout_rate", 0.1, 0.5)
    ext_neg_weight = trial.suggest_float("ext_neg_weight", 1.0, 3.0)
    ext_pos_weight = trial.suggest_float("ext_pos_weight", 1.0, 3.0)

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
        "dropout": dropout,
        "ext_neg_weight": ext_neg_weight,
        "ext_pos_weight": ext_pos_weight
    },
        name=f"{project_name}_{training_type}_trial_{trial.number}", reinit=True) # The name that will be saved in the W&B platform

    try:
        class_weights = torch.tensor([ext_neg_weight, 1.0, 1.0, 1.0, ext_pos_weight], dtype=torch.float).to(device)

        # Prepare datasets
        train_dataset, _, labels = data_preparation.prepare_dataset(tokenizer, max_length)

        #CHANGE AND ADD STRATIFIED K FOLD###
        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        fold_val_F1_scores = []
        best_model_across_folds = None  # for saving the best model across all folds
        best_val_F1_across_folds = -1.0
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset, labels)):
            train_subset = Subset(train_dataset, train_idx)
            eval_subset = Subset(train_dataset, val_idx)

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

        
            # Hugging Face TrainingArguments
            training_args = TrainingArguments(
                output_dir=f"./results/{project_name}_{training_type}_trial_{trial.number}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=20,
                weight_decay=weight_decay,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                report_to=["wandb"],
                save_total_limit=1,
                log_level="info"
                )

            # we initialize the Trainer with our custom WeightedLossTrainer
            trainer = WeightedLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_subset,
                eval_dataset=eval_subset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)], #adding our custom callback with training metrics
                class_weights=class_weights # our class weights
            )

            #training the model
            trainer.train()
    
            #evaluating the model on validation set
            metrics = trainer.evaluate()
            wandb.log({
                f"fold_{fold}/Validation Accuracy": metrics["eval_accuracy"],
                f"fold_{fold}/Validation F1": metrics["eval_f1"],
                f"fold_{fold}/Validation Precision": metrics["eval_precision"],
                f"fold_{fold}/Validation Recall": metrics["eval_recall"]
            })

            # save best model
            if metrics["eval_f1"] > best_val_F1_across_folds:
                    best_val_F1_across_folds = metrics["eval_f1"]
                    best_model_across_folds = model.state_dict()
                    best_model_fold = fold  # track which fold it was

        # Save best model after loop
        if best_model_across_folds is not None:
            os.makedirs(f"results/{project_name}", exist_ok=True)
            torch.save(best_model_across_folds, f"results/{project_name}/best_model_trial_{trial.number}_fold_{best_model_fold}.pt")
        wandb.log({"Best Validation F1 across folds": best_val_F1_across_folds})    

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        wandb.log({"crash/traceback": tb})
        wandb.alert(title="Trial crashed", text=tb[:1024])
        raise
    finally:
        wandb.finish()



    return best_val_F1_across_folds
