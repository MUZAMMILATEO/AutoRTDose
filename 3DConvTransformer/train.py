import torch
import torch.optim as optim
from tqdm import tqdm
import os 
import argparse
import random
import numpy as np
import logging 

# --- IMPORT NECESSARY MODULES ---
try:
    from Models.model import FCBFormer
except ImportError:
    print("Error: Could not import FCBFormer from 'Models.model'. Please ensure 'model.py' is saved.")
    exit(1)

try:
    from Data.dataloader import get_data_loaders
except ImportError:
    print("Error: Could not import get_data_loaders from 'Data/dataloader.py'. Please ensure the file is saved in 'Data/' directory.")
    exit(1)

try:
    # Assuming the updated loss file is named losses.py
    from Metrics.losses import DosePredictionLoss as CombinedLoss 
except ImportError:
    print("Error: Could not import CombinedLoss from 'Metrics/losses.py'. Please ensure the file is saved in 'Metrics/' directory.")
    exit(1)
    
# Placeholder for device management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def setup_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="3D Conv Transformer Training")
    # FIX: Setting default data_path to '..' (parent directory) to correctly locate 'overlaid-data'.
    parser.add_argument('--data_path', type=str, default='..', help='Path to the directory containing overlaid-data folder.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='./checkpoints', help='Directory to save logs and model checkpoints')
    return parser.parse_args()

def calculate_mae(output, target):
    """
    Calculates Mean Absolute Error (MAE) between prediction and target.
    This serves as our primary 'accuracy' metric for regression.
    """
    # Calculate MAE across all elements
    mae = torch.abs(output - target).mean()
    return mae.item()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Runs a single training epoch, returning loss and MAE."""
    model.train()
    total_loss = 0
    total_mae = 0
    count = 0
    
    # X=Input (3ch), Y=Target Dose (1ch), Masks=10 Structure Masks
    for X, Y, Masks in tqdm(dataloader, desc="Training"): 
        X, Y, Masks = X.to(device), Y.to(device), Masks.to(device)
        
        optimizer.zero_grad()
        output = model(X)
        
        # Calculate combined loss
        loss = criterion(output, Y, Masks) 
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate MAE (our 'accuracy' metric)
        mae = calculate_mae(output, Y)
        total_mae += mae
        count += 1
        
    return total_loss / count, total_mae / count # Return both loss and MAE

def validate_epoch(model, dataloader, criterion, device):
    """Runs a single validation epoch, returning loss and MAE."""
    model.eval()
    total_loss = 0
    total_mae = 0
    count = 0
    
    with torch.no_grad():
        for X, Y, Masks in tqdm(dataloader, desc="Validation"):
            X, Y, Masks = X.to(device), Y.to(device), Masks.to(device)
            output = model(X)
            
            # Calculate combined loss
            loss = criterion(output, Y, Masks)
            total_loss += loss.item()
            
            # Calculate MAE (our 'accuracy' metric)
            mae = calculate_mae(output, Y)
            total_mae += mae
            count += 1
            
    return total_loss / count, total_mae / count # Return both loss and MAE


def main():
    args = parse_args()
    setup_seed(args.seed)

    # --- Configuration ---
    IN_CHANNELS = 3    
    OUT_CHANNELS = 1   
    INPUT_SIZE = 128
    
    ROOT_DATA_DIR = args.data_path

    # --- CHECKPOINT CONFIGURATION ---
    CHECKPOINT_DIR = args.log_dir
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    
    # Ensure the checkpoint directory exists
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print(f"Checkpoint directory ensured: {os.path.abspath(CHECKPOINT_DIR)}")
    except Exception as e:
        print(f"Error creating checkpoint directory {CHECKPOINT_DIR}: {e}")
        return
    
    # --- Data Loading (using actual loader) ---
    try:
        train_loader, val_loader = get_data_loaders(
            ROOT_DATA_DIR, 
            args.batch_size, 
            num_workers=4 
        )
    except FileNotFoundError as e:
        print(f"Data loading error. Check ROOT_DATA_DIR ('{ROOT_DATA_DIR}') and data structure. Error: {e}")
        return

    # --- Model, Loss, and Optimizer Initialization ---
    model = FCBFormer(IN_CHANNELS, OUT_CHANNELS, INPUT_SIZE).to(device)
    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')
    metrics_history = [] # List to store per-epoch metrics

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Train
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f} | Training MAE: {train_mae:.4f}")

        # Validate
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f} | Validation MAE: {val_mae:.4f}")

        # Record metrics history
        metrics_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae,
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"Validation loss improved. Model saved to: {os.path.abspath(CHECKPOINT_PATH)}")
            except Exception as e:
                print(f"ERROR: Could not save model to {CHECKPOINT_PATH}. Reason: {e}")
        else:
            print("Validation loss did not improve.")

    # --- Save Training History to File ---
    history_file_path = os.path.join(CHECKPOINT_DIR, 'training_history.txt')
    try:
        with open(history_file_path, 'w') as f:
            # Write header
            f.write("Epoch\tTrain Loss\tTrain MAE\tVal Loss\tVal MAE\n")
            # Write data
            for m in metrics_history:
                f.write(f"{m['epoch']}\t{m['train_loss']:.6f}\t{m['train_mae']:.6f}\t{m['val_loss']:.6f}\t{m['val_mae']:.6f}\n")
        print(f"\nTraining history saved to: {os.path.abspath(history_file_path)}")
    except Exception as e:
        print(f"ERROR: Could not save training history to file. Reason: {e}")


if __name__ == '__main__':
    main()
