import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch import nn, optim
from transformers import AutoConfig
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import optuna
import wandb

import numpy as np
import os
from models import data_preparation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#as always, define early stopping
def early_stop_check(patience, best_val_F1_score, best_val_F1_epoch, current_val_F1_score, current_val_F1_epoch):
    early_stop_flag = False
    if current_val_F1_score > best_val_F1_score:
        best_val_F1_score = current_val_F1_score
        best_val_F1_epoch = current_val_F1_epoch
    else:
        if current_val_F1_epoch - best_val_F1_epoch > patience:
            early_stop_flag = True
    return best_val_F1_score, best_val_F1_epoch, early_stop_flag


def train_model_with_hyperparams(model, train_loader, val_loader, optimizer, criterion, epochs, patience, trial ,device, project_name, fold):
    best_val_F1_score = 0.0
    best_val_F1_epoch = 0
    early_stop_flag = False
    best_model_state = None

    for epoch in range(1, epochs + 1):

        model.train() # Enable training mode
        train_loss = 0.0
        total_train_samples = 0
        correct_train_predictions = 0

        for batch in train_loader: #Iterates over the train_loader, which is a DataLoader object containing batches of training data.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad() # Reset gradients
            outputs = model(input_ids, attention_mask=attention_mask) # Forward pass
            logits = outputs.logits # save the logits (the raw output of the model)
            loss = criterion(logits, labels) # Calculate loss

            loss.backward() # Backward pass
            optimizer.step() # Update weights using the optimizer 

            # Accumulate training loss and predictions
            train_loss += loss.item() * input_ids.size(0)
            total_train_samples += input_ids.size(0)
            correct_train_predictions += (logits.argmax(dim=1) == labels).sum().item()

        train_loss /= total_train_samples
        train_accuracy = correct_train_predictions / total_train_samples

        ###  Validation loop  ###
        model.eval() # Enable evaluation mode
        val_loss = 0.0
        total_val_samples = 0
        correct_val_predictions = 0

        all_val_labels = []
        all_val_preds = []

        with torch.no_grad(): # Disable gradient computation
            for batch in val_loader: # iterate on the val_loader's batches 
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

                val_loss += loss.item() * input_ids.size(0)
                total_val_samples += input_ids.size(0)
                correct_val_predictions += (logits.argmax(dim=1) == labels).sum().item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(logits.argmax(dim=1).cpu().numpy())

        # calculate metrics 
        val_loss /= total_val_samples
        val_accuracy = correct_val_predictions / total_val_samples
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')

        # Check for early stopping
        best_val_F1_score, best_val_F1_epoch, early_stop_flag = early_stop_check(patience, best_val_F1_score, best_val_F1_epoch, val_f1, epoch)

        # Save the best model under the best_model_state parameter
        if val_f1 == best_val_F1_score:
            best_model_state = model.state_dict()

        # Log metrics to Weights & Biases - THIS IS WHERE WE TRACK THE RESULTS AND THE PROCESS
        
        wandb.log({
            "fold": fold,
            "Epoch": epoch,
            f"fold_{fold}/Train Loss": train_loss,
            f"fold_{fold}/Train Accuracy": train_accuracy,
            f"fold_{fold}/Validation Loss": val_loss,
            f"fold_{fold}/Validation Accuracy": val_accuracy,
            f"fold_{fold}/Validation Precision": val_precision,
            f"fold_{fold}/Validation Recall": val_recall,
            f"fold_{fold}/Validation F1": val_f1
        }, step=epoch + fold * 1000)  # ensures unique step per fold

        if early_stop_flag:  # Checks whether the early stopping condition has been met, as indicated by the early_stop_flag
            break # Exits the training loop immediately if the early stopping condition is satisfied

    # #NOT SURE IF WE SHOULD CHANGE IT'S LOCATION
    # if best_model_state is not None: # Save the best model as a .pt file
    #     if not os.path.exists(f"results/{project_name}"):
    #         os.makedirs(f"results/{project_name}")
    #     torch.save(best_model_state, f"results/{project_name}/best_model_trial_{trial.number}_fold_{fold}.pt")

    return best_val_F1_score


# Objective Function for Optuna
def objective(trial, tokenizer, model_name, model_class, base_attr, project_name, training_type, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameter suggestions - WE SHOULD ADD MORE HYPERPARAMETERS HERE
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-4)
    patience = trial.suggest_int("patience", 7, 10)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3, step=1)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    #adding weight tuning for the extreme under-represented labels:
    ext_neg_weight = trial.suggest_float("ext_neg_weight", 1.0, 3.0)
    ext_pos_weight = trial.suggest_float("ext_pos_weight", 1.0, 3.0)

    wandb.init(
    project=project_name,
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
    name=f"{project_name}_{training_type}_trial_{trial.number}") # The name that will be saved in the W&B platform

    # loading and splitting data
    train_dataset, _, labels = data_preparation.prepare_dataset(tokenizer,max_length)
    
    #using stratified kfold for balanced splits:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_val_F1_scores = []
    best_model_across_folds = None # for saving the best model across all folds
    best_val_F1_across_folds = -1.0

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset, labels)):
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True) # insert into a DataLoader
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False) # insert into a DataLoader

        # build the model:
        config = AutoConfig.from_pretrained(
        model_name,
        num_labels=5,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout
        )
        model = model_class.from_pretrained(model_name, config=config).to(device)

        base_model = getattr(model, base_attr)
        for param in base_model.parameters():
            param.requires_grad = False
        for param in base_model.encoder.layer[-num_layers:].parameters():
            param.requires_grad = True

        #class weights for the loss function
        class_weights = torch.tensor([ext_neg_weight, 1.0, 1.0, 1.0, ext_pos_weight], dtype=torch.float).to(device)

        # Define optimizer and loss function
        criterion = nn.CrossEntropyLoss(weights=class_weights) #multiclass classification
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #maybe also try adamW

        # Initialize Weights & Biases - the values in the config are the properties of each trial.

        # Train the model and get the best F1 score:
        best_fold_val_F1 = train_model_with_hyperparams(model, train_loader, val_loader, optimizer, criterion, epochs=20, patience=patience, trial=trial,device=device,project_name=project_name, fold=fold)
        fold_val_F1_scores.append(best_fold_val_F1) #this is our most important metrix - the best validation across all folds

        # choose best model for saving best model across all folds:
        if best_fold_val_F1 > best_val_F1_across_folds:
            best_val_F1_across_folds = best_fold_val_F1
            best_model_across_folds = model.state_dict()

    #saving best_model
        #NOT SURE IF WE SHOULD CHANGE IT'S LOCATION
    if best_model_across_folds is not None: # Save the best model as a .pt file
        if not os.path.exists(f"results/{project_name}"):
            os.makedirs(f"results/{project_name}")
        torch.save(best_model_across_folds, f"results/{project_name}/best_model_trial_{trial.number}.pt")

    # Calculate the average validation loss across folds
    mean_val_F1 = np.mean(fold_val_F1_scores)

    wandb.log({ 
        "mean_val_F1": mean_val_F1})
    print(f"Mean Validation F1 across folds: {mean_val_F1}")
    wandb.finish() # Finish the Weights & Biases run
    
    return mean_val_F1 # Return best validation acc as the objective to maximize