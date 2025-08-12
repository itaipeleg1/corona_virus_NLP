from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import wandb
from collections import Counter
import os
from models import data_preparation
from transformers import TrainingArguments
import transformers

from transformers import TrainingArguments
import inspect

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
    }
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, label_smoothing=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = torch.nn.CrossEntropyLoss(
            weight=(self.class_weights.to(logits.device) if self.class_weights is not None else None),
            label_smoothing=self.label_smoothing
        )
        loss = loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

def objective_HF(trial, tokenizer, model_name, model_class, base_attr, project_name, training_type, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class.from_pretrained(model_name, num_labels=5).to(device)
    # Hyperparameter suggestions - WE SHOULD ADD MORE HYPERPARAMETERS HERE
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 3e-2)
    patience = trial.suggest_int("patience", 5,7)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 2, step=1)
    label_smooth = 0.1  # label smoothing factor

    # Freeze layers if base_attr is valid
    base_model = getattr(model, base_attr, None)
    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.encoder.layer[-num_layers:].parameters():
        param.requires_grad = True

    # Prepare datasets
    train_dataset, eval_dataset, _ = data_preparation.prepare_dataset(tokenizer, max_length)

    y_train = data_preparation.extract_labels(train_dataset)  # ints 0..K-1
    counts  = Counter(y_train)
    classes = sorted(counts.keys())
    N, K    = sum(counts.values()), len(classes)
    class_weights = torch.tensor([N/(K*counts[c]) for c in classes], dtype=torch.float)
    


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
    # ---- training args (mirror PyTorch choices) ----
    out_dir = f"./results/{project_name}_{training_type}_trial_{trial.number}"
    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,                  # keep only best
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  
        greater_is_better=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=20,
        weight_decay=weight_decay,
        warmup_ratio=0.1,
        logging_steps=10,
        report_to=["wandb"],
        max_grad_norm=1.0,              
        label_smoothing_factor=0.0           # we apply smoothing in custom loss instead
    )

    # ---- trainer: weighted loss + label smoothing (no sampler; just shuffle internally) ----
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],

    )

    trainer.train()
    metrics = trainer.evaluate()
    wandb.finish()
    trials_dir = f"./results/{project_name}/trials"
    os.makedirs(trials_dir, exist_ok=True)

    hf_dir = os.path.join(trials_dir, f"model_trial_{trial.number}_hf")
    trainer.save_model(hf_dir)  # config + model weights in HF format

    pt_path = os.path.join(trials_dir, f"model_trial_{trial.number}.pt")
    torch.save(trainer.model.state_dict(), pt_path)  # plain PyTorch weights

    pt_path_dict = os.path.join(trials_dir, f"model_trial_{trial.number}_dict.pt")
    model_dict = {
        "state_dict": trainer.model.state_dict(),
        "lr_rate": float(learning_rate),
        "best_acc": float(metrics.get("eval_accuracy", 0.0)),
        "best_f1":  float(metrics.get("eval_f1", 0.0)),
    }
    torch.save(model_dict, pt_path_dict)

    # record for Optuna's global-best picker
    trial.set_user_attr("model_path_raw",  pt_path)
    trial.set_user_attr("model_path_dict", pt_path_dict)
    trial.set_user_attr("best_acc", float(metrics.get("eval_accuracy", 0.0)))
    trial.set_user_attr("best_f1",  float(metrics.get("eval_f1", 0.0)))
    trial.set_user_attr("learning_rate", float(learning_rate))

    # return the optimization objective (accuracy to match metric_for_best_model)
    return float(metrics.get("eval_accuracy", 0.0))