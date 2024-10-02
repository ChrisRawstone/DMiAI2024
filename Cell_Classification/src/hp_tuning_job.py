import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random
import logging
import gc
from models.model import get_models, FocalLoss, get_model_parallel, LabelSmoothingLoss, BalancedBCELoss
from data.make_dataset import LoadTifDataset, get_transforms
from tqdm import tqdm
from threading import Lock
import optuna
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from src.utils import calculate_custom_score
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

wandb.login(key="c187178e0437c71d461606e312d20dc9f1c6794f")

# =============================================================================================
# 1. Setup and Configuration
# =============================================================================================

SEED = 0

def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Define device and check available GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()

# Setup logging early to capture GPU info
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
logging.info(f"Number of GPUs available: {num_gpus}")

# Create directories
os.makedirs('plots', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# =============================================================================================
## 2. Objective Function for Optuna
# =============================================================================================

NUM_EPOCHS = 100
PATIENCE_EPOCHS = 10

# Initialize global variables for top 5 trials and overall best
top_5_trials = []
top_5_lock = Lock()

overall_best_score = -np.inf  # Initialize to negative infinity
overall_best_trial = None

def update_top_5(trial_number, custom_score, model, model_info):
    """
    Updates the global top 5 trials list if the current trial's score is among the top 5.
    Also updates the overall best trial if applicable.

    Args:
        trial_number (int): The Optuna trial number.
        custom_score (float): The custom score of the trial.
        model (nn.Module): The trained model.
        model_info (dict): The hyperparameters and other info of the trial.
    """
    global top_5_trials, overall_best_score, overall_best_trial
    with top_5_lock:
        # Update Top 5 Trials
        if len(top_5_trials) < 5:
            top_5_trials.append({
                'trial_number': trial_number,
                'custom_score': custom_score,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'model_info': model_info
            })
            # Sort descendingly
            top_5_trials = sorted(top_5_trials, key=lambda x: x['custom_score'], reverse=True)
            logging.info(f"Trial {trial_number} added to top 5 with score {custom_score:.3f}.")
        else:
            # Check if current score is better than the lowest in top 5
            if custom_score > top_5_trials[-1]['custom_score']:
                removed_trial = top_5_trials.pop(-1)
                top_5_trials.append({
                    'trial_number': trial_number,
                    'custom_score': custom_score,
                    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'model_info': model_info
                })
                # Sort descendingly
                top_5_trials = sorted(top_5_trials, key=lambda x: x['custom_score'], reverse=True)
                logging.info(f"Trial {trial_number} with score {custom_score:.3f} added to top 5. Removed Trial {removed_trial['trial_number']} with score {removed_trial['custom_score']:.3f}.")

global best_custom_score
best_custom_score = 0

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization with wandb integration and early stopping.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.

    Returns:
        float: Validation custom score to maximize.
    """
    seed = trial.number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.cuda.empty_cache()  
    gc.collect()  
    global best_custom_score

    # ---------------------------
    # 1. Hyperparameter Sampling
    # ---------------------------
    model_name = trial.suggest_categorical('model_name', ['ViT16',"ViT32",
                                                          'EfficientNetB0', 'EfficientNetB4', 'MobileNetV3', 'DenseNet121', 
                                                          'ResNet101', 'ResNet18','ResNet50',
                                                          "SwinTransformer_B224", "SwinTransformer_B256", "SwinTransformer_L384", "SwinTransformer_H384"
                                                          ])
    
    if model_name in ['ViT16', "ViT32"]:
        img_size = 224  # Fixed for ViT
    elif model_name in ['SwinTransformer_B224']:
        img_size = 224
    elif model_name in ['SwinTransformer_B256']:
        img_size = 256
    elif model_name in ['SwinTransformer_L384', 'SwinTransformer_H384']:
        img_size = 384
    else:
        img_size = trial.suggest_categorical(
            'img_size',
            [224, 400, 600, 800, 1000])

    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    gamma = trial.suggest_float('gamma', 1.0, 3.0)
    alpha = trial.suggest_float('alpha', 0.1, 0.9)
    
    # Add loss_function as a hyperparameter
    loss_function = trial.suggest_categorical('loss_function', [
        'BCEWithLogitsLoss',
        'WeightedBCEWithLogitsLoss',
        'FocalLoss',
        'BalancedCrossEntropyLoss',
        'LabelSmoothingLoss'
    ])
    
    # ---------------------------
    # 2. Initialize wandb Run
    # ---------------------------
    num_epochs = NUM_EPOCHS  # Total number of epochs
    patience = PATIENCE_EPOCHS    # Early stopping patience
    
    wandb_config = {
        'model_name': model_name,
        'img_size': img_size,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'gamma': gamma,
        'alpha': alpha,
        'loss_function': loss_function,
        'num_epochs': num_epochs,
        'patience': patience}
    
    # Sample optimizer hyperparameters before the fold loop
    optimizer_name = trial.suggest_categorical('optimizer_name', ['AdamW', 'SGD', 'RMSprop'])

    # Add specific configurations based on the optimizer chosen
    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        optimizer_hyperparams = {
            'momentum': momentum,
            'weight_decay': weight_decay
        }
    elif optimizer_name in ['AdamW', 'Adam']:
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
        
    # Update wandb config with optimizer hyperparameters
    wandb_config.update({
        'optimizer_name': optimizer_name,
        'momentum': optimizer_hyperparams.get('momentum', None),
        'beta1': beta1 if optimizer_name in ['AdamW', 'Adam'] else None,
        'beta2': beta2 if optimizer_name in ['AdamW', 'Adam'] else None,
        'epsilon': optimizer_hyperparams.get('eps', None),
        'alpha_optim': optimizer_hyperparams.get('alpha', None)})

    wandb.init(
        project='Cell_Classification_HP',  # Replace with your wandb project name
        config=wandb_config,
        reinit=True,  # Allows multiple wandb runs in the same script
        name=f"trial_{trial.number}")

    # ---------------------------
    # 3. Load Dataset and Set Up Cross-Validation
    # ---------------------------
    
    train_image_dir = Path("data/training_val")
    train_csv_path = Path("data/training_val.csv")

    # Get transforms
    train_transform, val_transform = get_transforms(img_size)

    # Create the full dataset
    full_dataset = LoadTifDataset(
        image_dir=train_image_dir,
        csv_file_path=train_csv_path,
        transform=None  # Transforms will be set later
    )

    # Extract labels from the full dataset
    labels = full_dataset.labels_df.iloc[:, 1].values.astype(int)  # Assuming labels are in the second column

    # Compute class counts and pos_weight for Weighted BCE Loss
    class_counts = Counter(labels)
    num_positive = class_counts[1] if 1 in class_counts else 0
    num_negative = class_counts[0] if 0 in class_counts else 0
    if num_positive == 0 or num_negative == 0:
        pos_weight = 1.0  # Avoid division by zero; adjust as needed
    else:
        pos_weight = num_negative / num_positive

    # Get indices
    indices = np.arange(len(full_dataset))

    # Set up Stratified K-Fold cross-validation
    n_splits = 5  # Number of folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []

    # ---------------------------
    # 4. Cross-Validation Loop
    # ---------------------------
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(indices, labels)):
        logging.info(f"Fold {fold_idx + 1}/{n_splits}")

        # Subset the datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

        # Set transforms for each subset
        train_dataset.dataset.transform = train_transform
        val_dataset.dataset.transform = val_transform

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle is okay here since sampler is not used
            num_workers=8,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
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
        
        # Set up the loss function based on loss_function hyperparameter
        if loss_function == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss().to(device)
        elif loss_function == 'WeightedBCEWithLogitsLoss':
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
        elif loss_function == 'FocalLoss':
            criterion = FocalLoss(alpha=alpha, gamma=gamma, logits=True).to(device)
        elif loss_function == 'BalancedCrossEntropyLoss':
            criterion = BalancedBCELoss().to(device)
        elif loss_function == 'LabelSmoothingLoss':
            criterion = LabelSmoothingLoss(alpha).to(device)

        # Create optimizer using the sampled hyperparameters
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=optimizer_hyperparams['momentum'],
                weight_decay=optimizer_hyperparams['weight_decay']
            )
        elif optimizer_name in ['AdamW', 'Adam']:
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

        # running model with this configuration
        logging.info("###############################################")
        logging.info(f"Running model: {model_name} with Image Size: {img_size}, Batch Size: {batch_size}, LR: {lr}, Weight Decay: {weight_decay}, Gamma: {gamma}, Alpha: {alpha}, Loss Function: {loss_function}")
        logging.info(f"Optimizer: {optimizer_name}")
        logging.info("###############################################")

        # ---------------------------
        # 6. Training with Early Stopping
        # ---------------------------
        trial_best_score = -np.inf
        epochs_without_improvement = 0    
        best_epoch = 0
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs} - Fold {fold_idx + 1}/{n_splits}")

            # Training Phase
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []

            progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
            for images, labels in progress_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):  # Mixed precision
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

            # Validation Phase
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc='Validation', leave=False)
                for images, labels in progress_bar:
                    images = images.to(device, non_blocking=True)
                    labels = labels.float().unsqueeze(1).to(device, non_blocking=True)

                    with torch.amp.autocast("cuda"):  # Mixed precision
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)

                    preds = torch.sigmoid(outputs).detach().cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(labels.detach().cpu().numpy())

            avg_val_loss = val_loss / len(val_loader.dataset)

            # Calculate custom score
            preds_binary = (np.array(val_preds) > 0.5).astype(int)
            custom_score = calculate_custom_score(val_targets, preds_binary)
            recall_score_val = recall_score(val_targets, preds_binary, zero_division=0, average='macro')
            scheduler.step()
            
            # Log custom score
            logging.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Custom Score: {custom_score:.4f}, Recall Score: {recall_score_val:.4f}")

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
            
            # Early Stopping Logic
            if recall_score_val > trial_best_score:
                best_epoch = epoch + 1
                trial_best_score = recall_score_val
                epochs_without_improvement = 0
            else: 
                epochs_without_improvement += 1
            
            # Check if early stopping condition is met
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break
            
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
        'momentum': optimizer_hyperparams.get('momentum', None),
        'beta1': beta1 if optimizer_name in ['AdamW', 'Adam'] else None,
        'beta2': beta2 if optimizer_name in ['AdamW', 'Adam'] else None,
        'epsilon': optimizer_hyperparams.get('eps', None),
        'alpha_optim': optimizer_hyperparams.get('alpha', None),
        'loss_function': loss_function,
        'custom_score' : avg_custom_score, 
        'recall_score': recall_score_val}

    # Update best_custom_score if needed
    if avg_custom_score > best_custom_score:
        best_custom_score = avg_custom_score

        # Save the best model locally
        save_path_optuna = 'checkpoints/best_model_optuna.pth'
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), save_path_optuna)
        else:
            torch.save(model.state_dict(), save_path_optuna)
        
        logging.info(f"New best Optuna model saved with Val Score: {best_custom_score:.4f}")

        with open('checkpoints/model_info_optuna.json', 'w') as f:
            json.dump(model_info_optuna, f, indent=4)
        
    update_top_5(trial.number, avg_custom_score, model, model_info_optuna)

    wandb.log({'avg_custom_score': avg_custom_score})
    wandb.finish()

    torch.cuda.empty_cache()
    gc.collect()

    return avg_custom_score


if __name__ == "__main__":
    
    n_trials = 200
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=None)

    logging.info("===== Best Trial =====")
    best_trial = study.best_trial

    logging.info(f"  Value (Custom Score): {best_trial.value}")
    logging.info("  Params: ")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")
    
    # =============================================================================================
    # Save Top 5 Trials
    # =============================================================================================
    
    logging.info("===== Top 5 Trials =====")
    for i, trial in enumerate(top_5_trials, start=1):
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

    logging.info("Top 5 best models and their configurations have been saved successfully.")
