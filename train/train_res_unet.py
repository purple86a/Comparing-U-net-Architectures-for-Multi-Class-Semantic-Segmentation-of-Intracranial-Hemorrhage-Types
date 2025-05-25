import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import logging
import datetime
import csv
import argparse

from models.resunet_model import get_resnet_unet
from dataset.dataset_augmented import create_simple_data_loaders, get_dataset_paths, get_class_weights

# Set up logging
def setup_logging():
    """Set up the logging system"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up file name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_resUnet_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return log_file, timestamp

# Set up CSV logging for batch-level metrics
def setup_batch_logging(timestamp):
    """Set up CSV logging for batch-level metrics"""
    batch_log_file = f"logs/batch_metrics_resUnet_{timestamp}.csv"
    
    # Create CSV file with headers
    with open(batch_log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Phase', 'Epoch', 'Batch', 'Loss', 'Dice_BG', 'Dice_EDH', 'Dice_IPH', 'Dice_IVH', 'Dice_SAH', 'Dice_SDH', 'Mean_Dice'])
    
    return batch_log_file

# Define metrics for semantic segmentation
def calculate_iou(pred, target, n_classes=6):
    """Calculate IoU (Intersection over Union) for each class"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # For each class
    for cls in range(n_classes):
        # True Positive (TP): prediction is cls and ground truth is cls
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        
        if union.item() == 0:
            # If there is no ground truth of this class in this image, ignore it
            iou = torch.tensor(1.0)
        else:
            iou = intersection.float() / union.float()
        
        ious.append(iou.item())
    
    return ious

def calculate_dice(pred, target, n_classes=6):
    """Calculate Dice coefficient for each class"""
    dices = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    # For each class
    for cls in range(n_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum()
        cardinality = pred_cls.sum() + target_cls.sum()
        
        if cardinality.item() == 0:
            # If there is no ground truth and no prediction of this class, dice = 1
            dice = torch.tensor(1.0)
        else:
            dice = (2.0 * intersection) / cardinality
        
        dices.append(dice.item())
    
    return dices

def log_batch_metrics(batch_log_file, phase, epoch, batch_idx, loss, dices):
    """Log batch metrics to CSV file"""
    mean_dice = np.mean(dices)
    
    with open(batch_log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([phase, epoch, batch_idx, loss] + dices + [mean_dice])

def save_prediction_sample(images, masks, preds, sample_idx, class_names, phase=0, epoch=0):
    """Save visualization of predictions"""
    # Move tensors to CPU and convert to numpy
    image = images[0].cpu().numpy()[0]  # First image in batch, remove channel dim
    mask = masks[0].cpu().numpy()       # First mask in batch
    pred = preds[0].cpu().numpy()       # First prediction in batch
    
    # Create a colormap for visualization
    cmap = plt.cm.get_cmap('viridis', len(class_names))
    
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    plt.colorbar(ticks=range(len(class_names)), label='Class')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Plot prediction
    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    plt.colorbar(ticks=range(len(class_names)), label='Class')
    plt.title('Prediction')
    plt.axis('off')
    
    # Create directory if it doesn't exist
    os.makedirs('results_resUnet/predictions', exist_ok=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f'results_resUnet/predictions/phase{phase}_epoch{epoch}_sample{sample_idx}.png')
    plt.close()

def train():
    # Set up logging
    log_file, timestamp = setup_logging()
    batch_log_file = setup_batch_logging(timestamp)
    
    logging.info("Starting two-phase training with ResNet-101 UNet and augmented dataset")
    
    # Create output directory for results
    os.makedirs('results_resnet', exist_ok=True)
    
    # Create data loaders with simple data loader that doesn't apply augmentations
    train_loader, val_loader, test_loader = create_simple_data_loaders(batch_size=8, num_workers=0)
    
    # Get class weights for handling imbalance
    image_paths, mask_paths = get_dataset_paths()
    class_weights = get_class_weights(mask_paths)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Set class weights for loss function
    class_weights = class_weights.to(device)
    logging.info(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Define model
    model = get_resnet_unet(in_channels=1, out_channels=6, pretrained=True).to(device)  # 6 classes: background + 5 hemorrhage types
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Use weighted CrossEntropyLoss for multiclass segmentation
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    class_names = ['Background', 'EDH', 'IPH', 'IVH', 'SAH', 'SDH']
    
    # Two-phase training
    phase1_epochs = 5  # Freeze encoder and train decoder only
    phase2_epochs = 30  # Unfreeze encoder and train full model
    
    # Phase 1: Freeze encoder and train decoder only
    logging.info("Phase 1: Training decoder only (encoder frozen)")
    model.freeze_encoder()
    
    # Use higher learning rate for decoder only
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )
    
    # Phase 1 training
    for epoch in range(phase1_epochs):
        model.train()
        running_loss = 0.0
        epoch_dice_scores = np.zeros(6)
        batch_count = 0
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Phase 1 - Epoch {epoch+1}/{phase1_epochs}")
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)  # Shape: (B, 6, H, W)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Get predictions and calculate Dice score
            preds = torch.argmax(outputs, dim=1)
            batch_dices = calculate_dice(preds, masks)
            epoch_dice_scores += np.array(batch_dices)
            batch_count += 1
            
            # Log batch metrics
            log_batch_metrics(batch_log_file, 1, epoch+1, batch_idx+1, loss.item(), batch_dices)
            
            # Update progress bar
            running_loss += loss.item()
            mean_dice = np.mean(batch_dices)
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Dice": f"{mean_dice:.4f}"
            })
        
        # Average training metrics for epoch
        avg_train_loss = running_loss / len(train_loader)
        avg_train_dice = epoch_dice_scores / batch_count
        mean_train_dice = np.mean(avg_train_dice)
        
        train_losses.append(avg_train_loss)
        
        # Log epoch training metrics
        logging.info(f"Phase 1 - Epoch {epoch+1}/{phase1_epochs}")
        logging.info(f"  Train Loss: {avg_train_loss:.4f}, Mean Dice: {mean_train_dice:.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        class_ious = np.zeros(6)
        class_dices = np.zeros(6)
        val_samples = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                
                # Calculate IoU and Dice for each batch
                ious = calculate_iou(preds, masks)
                dices = calculate_dice(preds, masks)
                
                # Accumulate metrics
                class_ious += np.array(ious)
                class_dices += np.array(dices)
                val_samples += 1
                
                # Save a sample prediction
                if epoch == 0 and val_samples == 1:
                    save_prediction_sample(images, masks, preds, 0, class_names, phase=1, epoch=epoch)
        
        # Average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_class_ious = class_ious / val_samples
        avg_class_dices = class_dices / val_samples
        
        # Calculate mean metrics
        mean_iou = np.mean(avg_class_ious)
        mean_dice = np.mean(avg_class_dices)
        
        # Log validation metrics
        logging.info(f"  Val Loss: {avg_val_loss:.4f}, Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")
        logging.info("  Per-class validation metrics:")
        
        for i, class_name in enumerate(class_names):
            logging.info(f"    {class_name}: IoU={avg_class_ious[i]:.4f}, Dice={avg_class_dices[i]:.4f}")
        
        # Save best model from Phase 1
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'results_resUnet/best_phase1_resnet101_model.pth')
            logging.info("  New best Phase 1 model saved!")
    
    # Phase 2: Unfreeze encoder and train full model
    logging.info("Phase 2: Training full model (encoder unfrozen)")
    model.unfreeze_encoder()
    
    # Use lower learning rate for full model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Reset best validation loss for Phase 2
    best_val_loss = float('inf')
    
    # Phase 2 training
    for epoch in range(phase2_epochs):
        model.train()
        running_loss = 0.0
        epoch_dice_scores = np.zeros(6)
        batch_count = 0
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Phase 2 - Epoch {epoch+1}/{phase2_epochs}")
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)  # Shape: (B, 6, H, W)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Get predictions and calculate Dice score
            preds = torch.argmax(outputs, dim=1)
            batch_dices = calculate_dice(preds, masks)
            epoch_dice_scores += np.array(batch_dices)
            batch_count += 1
            
            # Log batch metrics
            log_batch_metrics(batch_log_file, 2, epoch+1, batch_idx+1, loss.item(), batch_dices)
            
            # Update progress bar
            running_loss += loss.item()
            mean_dice = np.mean(batch_dices)
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Dice": f"{mean_dice:.4f}"
            })
        
        # Average training metrics for epoch
        avg_train_loss = running_loss / len(train_loader)
        avg_train_dice = epoch_dice_scores / batch_count
        mean_train_dice = np.mean(avg_train_dice)
        
        train_losses.append(avg_train_loss)
        
        # Log epoch training metrics
        logging.info(f"Phase 2 - Epoch {epoch+1}/{phase2_epochs}")
        logging.info(f"  Train Loss: {avg_train_loss:.4f}, Mean Dice: {mean_train_dice:.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        class_ious = np.zeros(6)
        class_dices = np.zeros(6)
        val_samples = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                
                # Calculate IoU and Dice for each batch
                ious = calculate_iou(preds, masks)
                dices = calculate_dice(preds, masks)
                
                # Accumulate metrics
                class_ious += np.array(ious)
                class_dices += np.array(dices)
                val_samples += 1
                
                # Save a sample prediction
                if epoch % 5 == 0 and val_samples == 1:
                    save_prediction_sample(images, masks, preds, 0, class_names, phase=2, epoch=epoch)
        
        # Average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_class_ious = class_ious / val_samples
        avg_class_dices = class_dices / val_samples
        
        # Calculate mean metrics
        mean_iou = np.mean(avg_class_ious)
        mean_dice = np.mean(avg_class_dices)
        
        # Log validation metrics
        logging.info(f"  Val Loss: {avg_val_loss:.4f}, Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")
        logging.info("  Per-class validation metrics:")
        
        for i, class_name in enumerate(class_names):
            logging.info(f"    {class_name}: IoU={avg_class_ious[i]:.4f}, Dice={avg_class_dices[i]:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'results_resUnet/best_resnet101_model.pth')
            logging.info("  New best model saved!")
    
    # Save final model
    torch.save(model.state_dict(), 'results_resUnet/final_resnet101_model.pth')
    logging.info("Final model saved")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=phase1_epochs, color='r', linestyle='--', label='Phase 1 → Phase 2')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('ResNet-101 UNet: Training and Validation Loss')
    plt.legend()
    plt.savefig('results_resUnet/loss_curve.png')
    logging.info("Loss curve saved to results_resUnet/loss_curve.png")
    
    # Test the final model
    test_model(model, test_loader, device, class_names)
    
    logging.info("Training complete!")

def test_model(model, test_loader, device, class_names):
    """Evaluate model performance on test set"""
    logging.info("Evaluating model on test set")
    
    model.eval()
    class_ious = np.zeros(6)
    class_dices = np.zeros(6)
    test_samples = 0
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate metrics
            ious = calculate_iou(preds, masks)
            dices = calculate_dice(preds, masks)
            
            # Accumulate metrics
            class_ious += np.array(ious)
            class_dices += np.array(dices)
            test_samples += 1
            
            # Save a few example predictions for visualization
            if i < 5:
                save_prediction_sample(images, masks, preds, i, class_names, phase=3, epoch=0)
    
    # Average test metrics
    avg_class_ious = class_ious / test_samples
    avg_class_dices = class_dices / test_samples
    
    # Calculate mean metrics
    mean_iou = np.mean(avg_class_ious)
    mean_dice = np.mean(avg_class_dices)
    
    logging.info("\nTest Results:")
    logging.info(f"  Mean IoU: {mean_iou:.4f}")
    logging.info(f"  Mean Dice: {mean_dice:.4f}")
    
    # Log per-class metrics
    logging.info("  Per-class test metrics:")
    for i, class_name in enumerate(class_names):
        logging.info(f"    {class_name}: IoU={avg_class_ious[i]:.4f}, Dice={avg_class_dices[i]:.4f}")
    
    # Save results to CSV
    with open('results_resUnet/test_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'IoU', 'Dice'])
        for i, class_name in enumerate(class_names):
            writer.writerow([class_name, f"{avg_class_ious[i]:.4f}", f"{avg_class_dices[i]:.4f}"])
        writer.writerow(['Mean', f"{mean_iou:.4f}", f"{mean_dice:.4f}"])

if __name__ == "__main__":
    
    multiprocessing.freeze_support()

    train() 