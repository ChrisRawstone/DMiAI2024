import os
import gc
import json
import logging
import random
from collections import Counter
from functools import partial
from pathlib import Path
from threading import Lock

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.make_dataset import (LoadTifDataset, create_sampler,
                               get_dataloaders_final_train, get_transforms)
from models.model import (BalancedBCELoss, FocalLoss, get_model_parallel,
                          get_models)
from src.utils import calculate_custom_score

# =============================================================================================
# 1. Configuration and Setup
# =============================================================================================

SEED = 0
NUM_EPOCHS = 100
PATIENCE_EPOCHS = 10
N_TRIALS = 200
TOP_N_TRIALS = 5

def set_seed(seed=0):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TrialManager:
    """Manages the top N trials and the overall best trial during hyperparameter optimization."""

    def __init__(self, top_n=5):
        self.top_trials = []
        self.top_n = top_n
        self.lock = Lock()
        self.overall_best_score = -np.inf
        self.overall_best_trial = None

    def update_top_trials(self, trial_number, custom_score, model, model_info):
        """
        Updates the top N trials with the current trial if applicable.

        Args:
            trial_number (int): The Optuna trial number.
            custom_score (float): The custom score of the trial.
            model (nn.Module): The trained model.
            model_info (dict): The hyperparameters and other info of the trial.
        """
        with self.lock:
            # Update Top Trials
            if len(self.top_trials) < self.top_n:
                self.top_trials.append({
                    'trial_number': trial_number,
                    'custom_score': custom_score,
                    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'model_info': model_info
                })
                self.top_trials.sort(key=lambda x: x['custom_score'], reverse=True)
                logging.info(f"Trial {trial_number} added to top {self.top_n} with score {custom_score:.3f}.")
            else:
                # Check if current score is better than the lowest in top trials
                if custom_score > self.top_trials[-1]['custom_score']:
                    removed_trial = self.top_trials.pop(-1)
                    self.top_trials.append({
                        'trial_number': trial_number,
                        'custom_score': custom_score,
                        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                        'model_info': model_info
                    })
                    self.top_trials.sort(key=lambda x: x['custom_score'], reverse=True)
                    logging.info(f"Trial {trial_number} with score {custom_score:.3f} added to top {self.top_n}. Removed Trial {removed_trial['trial_number']} with score {removed_trial['custom_score']:.3f}.")

            # Update overall best score and trial
            if custom_score > self.overall_best_score:
                self.overall_best_score = custom_score
                self.overall_best_trial = trial_number


def train_loop(model, criterion, optimizer, scaler, train_loader, device):
    """Training loop for one epoch."""
    model.train()
    train_loss = 0
    train_preds = []
    train_targets = []

    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):  # Mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)

        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        train_preds.extend(preds)
        train_targets.extend(labels.detach().cpu().numpy())

        progress_bar.set_postfix({'Loss': loss.item()})

    avg_train_loss = train_loss / len(train_loader.dataset)
    return avg_train_loss, train_preds, train_targets


def eval_loop(model, criterion, val_loader, device):
    """Evaluation loop for validation or testing."""
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):  # Mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.detach().cpu().numpy())

    avg_val_loss = val_loss / len(val_loader.dataset)
    return avg_val_loss, val_preds, val_targets


def calc_perf_metrics(preds, targets):
    """Calculates custom and recall scores."""
    preds_binary = (np.array(preds) > 0.5).astype(int)
    custom_score = calculate_custom_score(targets, preds_binary)
    recall_score_val = recall_score(targets, preds_binary, zero_division=0, average='macro')
    return custom_score, recall_score_val


def main():
    # Set up logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/training.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Set seed and device
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    logging.info(f"Number of GPUs available: {num_gpus}")

    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Initialize TrialManager
    trial_manager = TrialManager(top_n=TOP_N_TRIALS)

    # Login to wandb
    # wandb.login()  

    # Define the objective function
    def objective(trial, trial_manager):
        seed = trial.number
        set_seed(seed)
        torch.cuda.empty_cache()
        gc.collect()

        # ---------------------------
        # 1. Hyperparameter Sampling
        # ---------------------------
        model_name = trial.suggest_categorical('model_name', ['EfficientNetB0', 'DenseNet121', 'ResNet101', 'EfficientNetB4'])
        img_size = trial.suggest_categorical('img_size', [800, 1000, 1200, 1400])
        batch_size = trial.suggest_categorical('batch_size', [4])
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        gamma = trial.suggest_float('gamma', 1.0, 3.0)
        alpha = trial.suggest_float('alpha', 0.1, 0.9)
        loss_function = trial.suggest_categorical('loss_function', [
            'BCEWithLogitsLoss',
            'WeightedBCEWithLogitsLoss',
            'FocalLoss',
            'BalancedCrossEntropyLoss'
        ])
        optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'AdamW', 'RMSprop'])

        # Optimizer hyperparameters
        optimizer_hyperparams = {}
        if optimizer_name in ['AdamW', 'Adam']:
            beta1 = trial.suggest_float('beta1', 0.8, 0.99)
            beta2 = trial.suggest_float('beta2', 0.9, 0.999)
            epsilon = trial.suggest_float('epsilon', 1e-8, 1e-6)
            optimizer_hyperparams = {
                'betas': (beta1, beta2),
                'eps': epsilon,
                'weight_decay': weight_decay
            }
        elif optimizer_name == 'RMSprop':
            alpha_optim = trial.suggest_float('alpha_optim', 0.9, 0.99)
            epsilon = trial.suggest_float('epsilon', 1e-8, 1e-6)
            optimizer_hyperparams = {
                'alpha': alpha_optim,
                'eps': epsilon,
                'weight_decay': weight_decay
            }

        # ---------------------------
        # 2. Initialize wandb Run
        # ---------------------------
        num_epochs = NUM_EPOCHS
        patience = PATIENCE_EPOCHS
        wandb_config = {
            'model_name': model_name,
            'img_size': img_size,
            'batch_size': batch_size,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'gamma': gamma,
            'alpha': alpha,
            'loss_function': loss_function,
            'optimizer_name': optimizer_name,
            'num_epochs': num_epochs,
            'patience': patience
        }
        wandb_config.update(optimizer_hyperparams)

        wandb.init(
            project='Cell_Classification_HP_Final',
            config=wandb_config,
            reinit=True,
            name=f"trial_{trial.number}"
        )

        # ---------------------------
        # 3. Load Dataset and Set Up Cross-Validation
        # ---------------------------
        train_image_dir = Path("data/training")
        train_csv_path = Path("data/training.csv")
        train_transform, val_transform = get_transforms(img_size)
        _, test_loader, _ = get_dataloaders_final_train(batch_size, img_size)
        full_dataset = LoadTifDataset(
            image_dir=train_image_dir,
            csv_file_path=train_csv_path,
            transform=None
        )
        labels = full_dataset.labels_df.iloc[:, 1].values.astype(int)
        class_counts = Counter(labels)
        num_positive = class_counts.get(1, 0)
        num_negative = class_counts.get(0, 0)
        pos_weight = num_negative / num_positive if num_positive else 1.0
        indices = np.arange(len(full_dataset))
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        # ---------------------------
        # 4. Cross-Validation Loop
        # ---------------------------
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(indices, labels)):
            logging.info(f"Fold {fold_idx + 1}/{n_splits}")

            # Subset the datasets
            train_subset = torch.utils.data.Subset(full_dataset, train_indices)
            val_subset = torch.utils.data.Subset(full_dataset, val_indices)

            # Set transforms for each subset
            train_subset.dataset.transform = train_transform
            val_subset.dataset.transform = val_transform

            train_labels = train_subset.dataset.labels_df.iloc[train_indices, 1].values.astype(int)
            train_sampler = create_sampler(train_labels)

            # Create DataLoaders
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=8,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )

            # ---------------------------
            # 5. Model and Optimizer Setup
            # ---------------------------
            models_dict = get_models()
            model = models_dict[model_name]
            model = get_model_parallel(model)
            model = model.to(device)

            # Loss function
            if loss_function == 'BCEWithLogitsLoss':
                criterion = nn.BCEWithLogitsLoss().to(device)
            elif loss_function == 'WeightedBCEWithLogitsLoss':
                pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
            elif loss_function == 'FocalLoss':
                criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True, reduce=True).to(device)
            elif loss_function == 'BalancedCrossEntropyLoss':
                criterion = BalancedBCELoss().to(device)

            # Optimizer
            if optimizer_name in ['AdamW', 'Adam']:
                optimizer_class = optim.AdamW if optimizer_name == 'AdamW' else optim.Adam
                optimizer = optimizer_class(
                    model.parameters(),
                    lr=lr,
                    betas=optimizer_hyperparams['betas'],
                    eps=optimizer_hyperparams['eps'],
                    weight_decay=optimizer_hyperparams['weight_decay']
                )
            elif optimizer_name == 'RMSprop':
                optimizer = optim.RMSprop(
                    model.parameters(),
                    lr=lr,
                    alpha=optimizer_hyperparams['alpha'],
                    eps=optimizer_hyperparams['eps'],
                    weight_decay=optimizer_hyperparams['weight_decay']
                )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            scaler = torch.amp.GradScaler("cuda")

            logging.info("###############################################")
            logging.info(f"Running model: {model_name} with Image Size: {img_size}, Batch Size: {batch_size}, LR: {lr}, Weight Decay: {weight_decay}, Gamma: {gamma}, Alpha: {alpha}, Loss Function: {loss_function}, Optimizer: {optimizer_name}")
            logging.info("###############################################")

            # ---------------------------
            # 6. Training with Early Stopping
            # ---------------------------
            trial_best_score = -np.inf
            trial_best_loss = float('inf')
            epochs_without_improvement = 0
            best_epoch = 0
            for epoch in range(num_epochs):
                logging.info(f"Epoch {epoch+1}/{num_epochs} - Fold {fold_idx + 1}/{n_splits}")

                # Training Phase
                avg_train_loss, train_preds, train_targets = train_loop(model, criterion, optimizer, scaler, train_loader, device)

                # Validation Phase
                avg_val_loss, val_preds, val_targets = eval_loop(model, criterion, val_loader, device)

                # Test Phase
                avg_test_loss, test_preds, test_targets = eval_loop(model, criterion, test_loader, device)

                # Calculate performance
                custom_score_train, recall_score_train = calc_perf_metrics(train_preds, train_targets)
                custom_score, recall_score_val = calc_perf_metrics(val_preds, val_targets)
                custom_score_test, recall_score_test = calc_perf_metrics(test_preds, test_targets)

                scheduler.step()

                # Log score
                logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                logging.info(f"Custom Score TRAIN: {custom_score_train:.4f}, Recall Score Train: {recall_score_train:.4f}")
                logging.info(f"Custom Score VAL : {custom_score:.4f}, Recall Score: {recall_score_val:.4f}")
                logging.info(f"Custom Score TEST: {custom_score_test:.4f}, Recall Score: {recall_score_test:.4f}")

                # ---------------------------
                # 7. Logging Metrics to wandb
                # ---------------------------
                wandb.log({
                    'fold': fold_idx + 1,
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'custom_score': custom_score,
                    'recall_score': recall_score_val,
                    'learning_rate': scheduler.get_last_lr()[0]})

                # Early Stopping Logic Based on Validation Loss
                if avg_val_loss < trial_best_loss:
                    best_epoch = epoch + 1
                    trial_best_loss = avg_val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    logging.info(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

                if epochs_without_improvement >= patience:
                    logging.info(f"Early stopping triggered after {patience} epochs without improvement.")
                    break  # Exit the training loop

                # Optuna Trial Reporting and Pruning
                trial.report(trial_best_score, epoch)
                if trial.should_prune():
                    logging.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
                    raise optuna.exceptions.TrialPruned()

            # Record the best score for this fold
            fold_metrics.append(trial_best_score)
            logging.info(f"Best score for Fold {fold_idx + 1}: {trial_best_score:.4f} at epoch {best_epoch}")

            # Clean up
            torch.cuda.empty_cache()
            gc.collect()

        # ---------------------------
        # 8. Calculate Average Score Across Folds
        # ---------------------------
        avg_custom_score = np.mean(fold_metrics)
        logging.info(f"Average custom score across folds: {avg_custom_score}")

        model_info_optuna = {
            'model_name': model_name,
            'img_size': img_size,
            'batch_size': batch_size,
            'num_epochs': epoch + 1,
            'learning_rate': lr,
            'weight_decay': weight_decay,
            'gamma': gamma,
            'alpha': alpha,
            'Optimizer': optimizer_name,
            'loss_function': loss_function,
            'custom_score': avg_custom_score,
            'recall_score': recall_score_val
        }
        model_info_optuna.update(optimizer_hyperparams)

        # Final Evaluation
        avg_val_loss, val_preds, val_targets = eval_loop(model, criterion, val_loader, device)
        _, test_preds, test_targets = eval_loop(model, criterion, test_loader, device)
        custom_score, recall_score_val = calc_perf_metrics(val_preds, val_targets)
        custom_score_test, recall_score_test = calc_perf_metrics(test_preds, test_targets)
        logging.info(f"Final Custom Score VAL : {custom_score:.4f}, Recall Score: {recall_score_val:.4f}")
        logging.info(f"Final Custom Score TEST: {custom_score_test:.4f}, Recall Score: {recall_score_test:.4f}")

        # Update TrialManager with the current trial
        trial_manager.update_top_trials(trial.number, avg_custom_score, model, model_info_optuna)

        wandb.log({'avg_custom_score': avg_custom_score})
        wandb.finish()

        torch.cuda.empty_cache()
        gc.collect()

        return avg_custom_score

    # Optimize using Optuna
    objective_fn = partial(objective, trial_manager=trial_manager)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_fn, n_trials=N_TRIALS, timeout=None)

    # Logging Best Trial
    logging.info("===== Best Trial =====")
    best_trial = study.best_trial
    logging.info(f"  Value (Custom Score): {best_trial.value}")
    logging.info("  Params: ")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")

    # Save Top N Trials
    logging.info(f"===== Top {TOP_N_TRIALS} Trials =====")
    for i, trial in enumerate(trial_manager.top_trials, start=1):
        score = round(trial['custom_score'], 3)
        logging.info(f"Top {i}: Trial {trial['trial_number']} with Custom Score {score:.3f}")

        # Define filenames
        new_model_path = f'checkpoints/best_model_{i}_{score:.3f}.pth'
        new_json_path = f'checkpoints/best_model_{i}_{score:.3f}.json'

        # Save the model
        torch.save(trial['model_state_dict'], new_model_path)
        logging.info(f"Saved model for Trial {trial['trial_number']} as {new_model_path}")

        # Save the JSON configuration
        with open(new_json_path, 'w') as f:
            json.dump(trial['model_info'], f, indent=4)
        logging.info(f"Saved model info for Trial {trial['trial_number']} as {new_json_path}")

    logging.info(f"Top {TOP_N_TRIALS} best models and their configurations have been saved successfully.")


if __name__ == "__main__":
    main()