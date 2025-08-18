from zipfile import Path
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os
from models.data_preparation import prepare_dataset
import torch.nn.functional as F
from .model_loading import load_student_model
from pathlib import Path
from config import COMPRESSION_OUTPUT_DIR
import json

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=3.0, alpha=0.7, **kwargs): #setting deafult temperature and alpha
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.eval().to(self.args.device)
        for param in self.teacher.parameters(): #make sure all parameters don't require grad
            param.requires_grad = False
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = inputs["input_ids"].device
        # #making sure both are on same device:
        if next(self.teacher.parameters()).device != device:
            self.teacher.to(device)

        #forward pass
        outputs_student = model(**inputs)
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        loss_ce = F.cross_entropy(outputs_student.logits, inputs["labels"])
        loss_kl = F.kl_div(
            F.log_softmax(outputs_student.logits / self.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.temperature, dim=-1),
            reduction="batchmean") * (self.temperature ** 2)
        loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kl
        
        return (loss, outputs_student) if return_outputs else loss
    
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

#excepts student_model, student_tokenizer, original_model
def knowledge_distillation(student_key, model_key, student_model, student_tokenizer, original_model, do_train_distill: bool, temperature: float, alpha: float, epochs: int, output_dir=COMPRESSION_OUTPUT_DIR):
    summary_path = Path(output_dir) / model_key / f"{student_key}_knowledge_distillation_summary.json"
    training_summary = None

    if do_train_distill:
        print("\n=== STARTING DISTILLATION ===")
        #prepare datasets
        train_ds, val_ds, _ = prepare_dataset(student_tokenizer, max_length=128) #using same tokenizer 

        # define training arguments (simpler than original, few epochs)
        training_args = TrainingArguments(
            output_dir= f"./results/{student_key}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=20,
            save_total_limit=1,
            report_to=[]  # Disable W&B
        )
        trainer_distill = DistillationTrainer(
            model=student_model,
            teacher_model=original_model,
            temperature=temperature,
            alpha=alpha,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics
        )


        trainer_distill.train()
        print("\nDistillation complete. Student model trained.")
        print("Student model size:", sum(p.numel() for p in student_model.parameters()))

        trained_model = trainer_distill.model
        state_dict = trained_model.state_dict()
        accuracy = trainer_distill.evaluate()["eval_accuracy"]
        training_summary = {
            "model_key": model_key,
            "student_key": student_key,
            "learning_rate": training_args.learning_rate,
            "epochs": training_args.num_train_epochs,
            "accuracy": accuracy
        }
        # saving training summary (model saving is handled outside for all models)
        with open(summary_path, "w") as f:
            json.dump(training_summary, f, indent=4)

    else:
        # load trained student model and summary (r maybe just summary? then i  should save it)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trained_model, _ = load_student_model(student_key=student_key, num_labels=5, load_trained=True, device=device)
        if summary_path.exists():
            with open(summary_path, "r") as f:
                training_summary = json.load(f)
        else:
            print(f"⚠️ No training summary found at {summary_path}. Returning None.")
            training_summary = None
    return trained_model, training_summary

