import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import optuna
import wandb

import os
from models import data_preparation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#as always, define early stopping
def early_stop_check(patience, best_val_accuracy, best_val_accuracy_epoch, current_val_accuracy, current_val_accuracy_epoch):
    early_stop_flag = False
    if current_val_accuracy > best_val_accuracy:
        best_val_accuracy = current_val_accuracy
        best_val_accuracy_epoch = current_val_accuracy_epoch
    else:
        if current_val_accuracy_epoch - best_val_accuracy_epoch > patience:
            early_stop_flag = True
    return best_val_accuracy, best_val_accuracy_epoch, early_stop_flag


def train_model_with_hyperparams(model, train_loader, val_loader, optimizer, criterion, epochs, patience, trial,device):
    best_val_accuracy = 0.0
    best_val_accuracy_epoch = 0
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
        best_val_accuracy, best_val_accuracy_epoch, early_stop_flag = early_stop_check(patience, best_val_accuracy, best_val_accuracy_epoch, val_accuracy, epoch)

        # Save the best model under the best_model_state parameter
        if val_accuracy == best_val_accuracy:
            best_model_state = model.state_dict()

        # Log metrics to Weights & Biases - THIS IS WHERE WE TRACK THE RESULTS AND THE PROCESS
        
        wandb.log({ #log == logging of the training process (e.g. results) - will be done each epoch
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy,
            "Validation Precision": val_precision,
            "Validation Recall": val_recall,
            "Validation F1": val_f1})

        if early_stop_flag:  # Checks whether the early stopping condition has been met, as indicated by the early_stop_flag
            break # Exits the training loop immediately if the early stopping condition is satisfied

    #NOT SURE IF WE SHOULD CHANGE IT'S LOCATION
    if best_model_state is not None: # Save the best model as a .pt file
        torch.save(best_model_state, f"best_model_trial_{trial.number}.pt")

    return best_val_accuracy


# Objective Function for Optuna
def objective(trial, tokenizer, model_name, model, base_attr, project_name, training_type,max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameter suggestions - WE SHOULD ADD MORE HYPERPARAMETERS HERE
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-4)
    patience = trial.suggest_int("patience", 7, 10)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3, step=1)
    
    # Create the dataset and dataloaders

    train_dataset, eval_dataset,_ = data_preparation.prepare_dataset(tokenizer,max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # insert into a DataLoader
    val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False) # insert into a DataLoader

    model = model.to(device) # we initialize every model outside the objective function

    base_model = getattr(model, base_attr, None)
    if base_model is None:
        raise ValueError(f"Model does not have base attribute '{base_attr}'")

    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.encoder.layer[-num_layers:].parameters():
        param.requires_grad = True

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss() #multiclass classification
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) #maybe also try adamW

    # Initialize Weights & Biases - the values in the config are the properties of each trial.

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

    # Train the model and get the best validation accuracy
    best_val_accuracy = train_model_with_hyperparams(model, train_loader, val_loader, optimizer, criterion, epochs=20, patience=patience, trial=trial,device=device)

    wandb.finish() # Finish the Weights & Biases run
    
    return best_val_accuracy # Return best validation acc as the objective to maximize