#This program performs semantic segmentation on drone imagery using a VGG16-based U-Net architecture with PyTorch.

import os
import sys
import cv2
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms
from torchvision.models import VGG16_Weights


#This function sets up the device (GPU or CPU) and configures safety settings to prevent memory issues.
def setup_device_and_safety():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    #Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #GPU safety features to prevent overload and crashes.
    try:
        if torch.cuda.is_available():
            #Limit GPU memory growth to prevent OOM errors.
            torch.cuda.empty_cache()

            #Set memory allocation strategy to be more conservative.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            #Optional CUDA memory debugging configuration.
            #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

            #Set cudnn to use less memory.
            os.environ["CUDNN_AUTOTUNE_THRESHOLD"] = "0"

            print("\nGPU Safety Features Enabled.")
            print(f"Initial GPU memory allocated (MB): {torch.cuda.memory_allocated() / (1024**2):.2f}")
            print(f"Initial GPU memory reserved (MB): {torch.cuda.memory_reserved() / (1024**2):.2f}")

        #Test GPU with a small operation.
        if device.type == 'cuda':
            a = torch.randn(1024, 1024, device=device)
            b = torch.randn(1024, 1024, device=device)
            _ = torch.mm(a, b)
    except Exception as e:
        print(f"GPU safety setup encountered an issue: {e}")
        device = torch.device('cpu')
        print("Falling back to CPU.")

    #Set random seeds for reproducibility.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)  #For multi-GPU setups.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return device


#Data augmentation transform for visualization/demo.
transform = A.Compose([
    A.RandomCrop(width=4500, height=3000, p=1.0),
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.Rotate(limit=[60, 240], p=1.0, interpolation=cv2.INTER_NEAREST),
    A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.4], contrast_limit=0.2, p=1.0),
    A.OneOf([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, interpolation=cv2.INTER_NEAREST, p=0.5),
    ], p=1.0),
], p=1.0)


#This function visualizes images and their corresponding segmentation masks for comparison.
def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 16

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(16, 16), squeeze=True)
        f.set_tight_layout(h_pad=5, w_pad=5)

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 16), squeeze=True)
        plt.tight_layout(pad=0.2, w_pad=1.0, h_pad=0.01)

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original Image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original Mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed Image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed Mask', fontsize=fontsize)

    os.makedirs('./visuals', exist_ok=True)
    plt.savefig('./visuals/sample_augmented_comparison.png', facecolor='w', transparent=False, bbox_inches='tight', dpi=100)
    plt.close()


#This function loads the class dictionary mapping class names to RGB colors from the dataset.
def load_class_dict():
    class_dict_path = './semantic_drone_dataset/training_set/gt/semantic/class_dict.csv'
    class_dict_df = pd.read_csv(class_dict_path, index_col=False, skipinitialspace=True)

    label_names = list(class_dict_df['name'])
    label_codes = []
    r = np.asarray(class_dict_df['r'])
    g = np.asarray(class_dict_df['g'])
    b = np.asarray(class_dict_df['b'])

    for i in range(len(class_dict_df)):
        label_codes.append(tuple([r[i], g[i], b[i]]))

    code2id = {v: k for k, v in enumerate(label_codes)}
    id2code = {k: v for k, v in enumerate(label_codes)}
    name2id = {v: k for k, v in enumerate(label_names)}
    id2name = {k: v for k, v in enumerate(label_names)}

    print("First 5 label codes:", label_codes[:5])
    print("First 5 label names:", label_names[:5])
    print(f"Total number of classes: {len(label_codes)}")
    return label_names, label_codes, code2id, id2code, name2id, id2name


#This function converts RGB mask images to one-hot encoded format for training.
def rgb_to_onehot(rgb_image, colormap):
    """One hot encode RGB mask labels."""
    num_classes = len(colormap)
    shape = rgb_image.shape[:2] + (num_classes,)
    encoded_image = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(colormap):
        encoded_image[:, :, i] = np.all(rgb_image.reshape((-1, 3)) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image


#This function converts one-hot encoded masks back to RGB format for visualization.
def onehot_to_rgb(onehot, colormap):
    """Decode one-hot mask to RGB."""
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in colormap.keys():
        output[single_layer == k] = colormap[k]
    return np.uint8(output)


#This class implements a PyTorch Dataset for loading and preprocessing drone images and segmentation masks.
class DroneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, id2code, target_size=(512, 512), augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.target_size = target_size
        self.augment = augment
        self.id2code = id2code

        #Get all image filenames.
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')])

        #Image normalization transform.
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        #Load image.
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Load mask.
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        #Resize.
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        #Convert mask to one-hot encoding.
        mask_encoded = rgb_to_onehot(mask, self.id2code)

        #Convert to tensors.
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  #(H, W, C) -> (C, H, W).
        image = self.normalize(image)

        mask_encoded = torch.from_numpy(mask_encoded).permute(2, 0, 1).float()  #(H, W, C) -> (C, H, W).

        return image, mask_encoded


#This class implements a VGG16-based U-Net architecture for semantic segmentation.
class VGG16UNet(nn.Module):
    def __init__(self, num_classes=24, pretrained=True):
        super(VGG16UNet, self).__init__()

        #Load pretrained VGG16.
        if pretrained:
            vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg16 = models.vgg16(weights=None)

        #Encoder (VGG16 feature extractor).
        features = list(vgg16.features.children())

        #Block 1.
        self.block1 = nn.Sequential(*features[0:4])  #Conv1_1, ReLU, Conv1_2, ReLU.
        self.pool1 = features[4]  #MaxPool.

        #Block 2.
        self.block2 = nn.Sequential(*features[5:9])  #Conv2_1, ReLU, Conv2_2, ReLU.
        self.pool2 = features[9]  #MaxPool.

        #Block 3.
        self.block3 = nn.Sequential(*features[10:16])  #Conv3_1, ReLU, Conv3_2, ReLU, Conv3_3, ReLU.
        self.pool3 = features[16]  #MaxPool.

        #Block 4.
        self.block4 = nn.Sequential(*features[17:23])  #Conv4_1, ReLU, Conv4_2, ReLU, Conv4_3, ReLU.
        self.pool4 = features[23]  #MaxPool.

        #Block 5.
        self.block5 = nn.Sequential(*features[24:30])  #Conv5_1, ReLU, Conv5_2, ReLU, Conv5_3, ReLU.

        #Decoder.
        #UP 1.
        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.bn_up1 = nn.BatchNorm2d(512)
        self.conv_up1_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn_up1_1 = nn.BatchNorm2d(512)
        self.conv_up1_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_up1_2 = nn.BatchNorm2d(512)

        #UP 2.
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.bn_up2 = nn.BatchNorm2d(256)
        self.conv_up2_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn_up2_1 = nn.BatchNorm2d(256)
        self.conv_up2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_up2_2 = nn.BatchNorm2d(256)

        #UP 3.
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.bn_up3 = nn.BatchNorm2d(128)
        self.conv_up3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn_up3_1 = nn.BatchNorm2d(128)
        self.conv_up3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_up3_2 = nn.BatchNorm2d(128)

        #UP 4.
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn_up4 = nn.BatchNorm2d(64)
        self.conv_up4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_up4_1 = nn.BatchNorm2d(64)
        self.conv_up4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_up4_2 = nn.BatchNorm2d(64)

        #Final convolution.
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        #Encoder.
        block1_out = self.block1(x)
        x = self.pool1(block1_out)

        block2_out = self.block2(x)
        x = self.pool2(block2_out)

        block3_out = self.block3(x)
        x = self.pool3(block3_out)

        block4_out = self.block4(x)
        x = self.pool4(block4_out)

        x = self.block5(x)

        #Decoder.
        #UP 1.
        x = F.relu(self.bn_up1(self.upconv1(x)))
        x = torch.cat([x, block4_out], dim=1)
        x = F.relu(self.bn_up1_1(self.conv_up1_1(x)))
        x = F.relu(self.bn_up1_2(self.conv_up1_2(x)))

        #UP 2.
        x = F.relu(self.bn_up2(self.upconv2(x)))
        x = torch.cat([x, block3_out], dim=1)
        x = F.relu(self.bn_up2_1(self.conv_up2_1(x)))
        x = F.relu(self.bn_up2_2(self.conv_up2_2(x)))

        #UP 3.
        x = F.relu(self.bn_up3(self.upconv3(x)))
        x = torch.cat([x, block2_out], dim=1)
        x = F.relu(self.bn_up3_1(self.conv_up3_1(x)))
        x = F.relu(self.bn_up3_2(self.conv_up3_2(x)))

        #UP 4.
        x = F.relu(self.bn_up4(self.upconv4(x)))
        x = torch.cat([x, block1_out], dim=1)
        x = F.relu(self.bn_up4_1(self.conv_up4_1(x)))
        x = F.relu(self.bn_up4_2(self.conv_up4_2(x)))

        #Final convolution.
        x = self.final_conv(x)

        return x


#This function calculates the Dice coefficient metric for evaluating segmentation quality.
def dice_coefficient(y_pred, y_true, smooth=1.0):
    """Calculate Dice coefficient."""
    y_pred_f = y_pred.flatten()
    y_true_f = y_true.flatten()
    intersection = (y_pred_f * y_true_f).sum()
    return (2. * intersection + smooth) / (y_pred_f.sum() + y_true_f.sum() + smooth)


#This function prepares training and validation data loaders with proper augmentation and batching.
def prepare_dataloaders(id2code, device):
    #Update these paths according to your dataset structure.
    train_images_dir = './semantic_drone_dataset/training_set/images/'
    train_masks_dir = './semantic_drone_dataset/training_set/gt/semantic/label_images/'

    #Create datasets.
    full_dataset = DroneDataset(train_images_dir, train_masks_dir, id2code, target_size=(512, 512))

    #Split into train and validation (80-20 split).
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    #Original hyperparameters from published results.
    batch_size = 8  #Original batch size from paper achieving ~0.87 validation dice.
    print(f"\nUsing batch size: {batch_size} (original configuration)")

    #Create data loaders with pinned memory for faster GPU transfer.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  #Set to 0 to avoid multiprocessing issues on Windows.
        pin_memory=torch.cuda.is_available()  #Faster GPU transfer.
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()  #Faster GPU transfer.
    )

    print(f"\nNumber of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    #Clear GPU cache before training.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU memory after data loader setup:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    return train_loader, val_loader


#This function initializes the model, optimizer, loss function, and learning rate scheduler.
def init_model_and_optim(device):
    #Initialize model.
    model = VGG16UNet(num_classes=24, pretrained=True).to(device)

    #Loss function.
    criterion = nn.CrossEntropyLoss()

    #Optimizer.
    initial_lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    #Learning rate scheduler (exponential decay).
    def lr_lambda(epoch):
        return 0.1 ** (epoch / 20)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(f"Model initialized on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model, criterion, optimizer, scheduler


#This function trains the model for one epoch with automatic GPU memory management and error recovery.
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):
        try:
            images = images.to(device)
            masks = masks.to(device)

            #Forward pass.
            outputs = model(images)

            #Calculate loss.
            loss = criterion(outputs, masks)

            #Backward pass and optimization.
            optimizer.zero_grad()
            loss.backward()

            #Gradient clipping to prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            #Calculate dice coefficient.
            with torch.no_grad():
                outputs_softmax = F.softmax(outputs, dim=1)
                dice = dice_coefficient(outputs_softmax, masks)

            running_loss += loss.item()
            running_dice += dice.item()

            #GPU memory management - clear cache periodically.
            if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Dice: {dice.item():.4f}")
                if torch.cuda.is_available():
                    print(f"    GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

            #Free up memory.
            del images, masks, outputs, loss

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nGPU OUT OF MEMORY at batch {batch_idx + 1}.")
                print("Clearing cache and skipping this batch.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

    epoch_loss = running_loss / len(train_loader)
    epoch_dice = running_dice / len(train_loader)

    return epoch_loss, epoch_dice


#This function evaluates the model on validation data with automatic GPU memory management.
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            try:
                images = images.to(device)
                masks = masks.to(device)

                #Forward pass.
                outputs = model(images)

                #Calculate loss.
                loss = criterion(outputs, masks)

                #Calculate dice coefficient.
                outputs_softmax = F.softmax(outputs, dim=1)
                dice = dice_coefficient(outputs_softmax, masks)

                running_loss += loss.item()
                running_dice += dice.item()

                #Free up memory.
                del images, masks, outputs, loss

                #Clear GPU cache periodically.
                if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nGPU OUT OF MEMORY during validation at batch {batch_idx + 1}.")
                    print("Clearing cache and skipping this batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    epoch_loss = running_loss / len(val_loader)
    epoch_dice = running_dice / len(val_loader)

    return epoch_loss, epoch_dice


#This function orchestrates the complete training loop with early stopping and best model checkpointing.
def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20, patience=6):
    #Lists to store metrics.
    train_losses = []
    train_dices = []
    val_losses = []
    val_dices = []
    learning_rates = []

    #Create directory for saving models.
    os.makedirs('models', exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...\n")

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f"Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}")

        #Train.
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_dices.append(train_dice)

        #Validate.
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        #Save best model.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_dice': train_dice,
                'val_dice': val_dice,
            }, 'models/vgg16_unet_best.pth')
            print(f"Model saved. (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        #Early stopping.
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        #Step scheduler.
        scheduler.step()

    print("\nTraining completed.")

    return train_losses, train_dices, val_losses, val_dices, learning_rates


#This function saves training history and generates performance visualization plots.
def save_history_and_plot(train_losses, train_dices, val_losses, val_dices, learning_rates):
    history = {
        'train_loss': train_losses,
        'train_dice': train_dices,
        'val_loss': val_losses,
        'val_dice': val_dices,
        'learning_rate': learning_rates
    }

    with open('trainHistoryDict.pkl', 'wb') as file:
        pickle.dump(history, file)

    #Also save as CSV.
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'train_dice_coef': train_dices,
        'val_loss': val_losses,
        'val_dice_coef': val_dices,
        'lr': learning_rates
    })
    history_df.to_csv('model_training_csv.log', index=False)
    print("Training history saved.")

    #Plot metrics.
    os.makedirs('./visuals', exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(30, 5))
    ax = ax.ravel()

    #Dice.
    ax[0].plot(train_dices, 'o-', label='Train')
    ax[0].plot(val_dices, 'o-', label='Validation')
    ax[0].set_title('Dice Coefficient vs Epochs', fontsize=16)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Dice Coefficient')
    ax[0].set_xticks(np.arange(0, max(1, len(train_dices)) + 1, 2))
    ax[0].legend()
    ax[0].grid(True, color="lightgray", linewidth="0.8", linestyle="-")

    #Loss.
    ax[1].plot(train_losses, 'o-', label='Train')
    ax[1].plot(val_losses, 'o-', label='Validation')
    ax[1].set_title('Loss vs Epochs', fontsize=16)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].set_xticks(np.arange(0, max(1, len(train_losses)) + 1, 2))
    ax[1].legend()
    ax[1].grid(True, color="lightgray", linewidth="0.8", linestyle="-")

    #LR.
    ax[2].plot(learning_rates, 'o-')
    ax[2].set_title('Learning Rate vs Epochs', fontsize=16)
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Learning Rate')
    ax[2].set_xticks(np.arange(0, max(1, len(learning_rates)) + 1, 2))
    ax[2].grid(True, color="lightgray", linewidth="0.8", linestyle="-")

    plt.savefig('./visuals/model_metrics_plot.png', facecolor='w', transparent=False, bbox_inches='tight', dpi=150)
    plt.close()


#This function generates segmentation predictions and saves sample results for visual evaluation.
def generate_predictions_and_samples(model, val_loader, id2code, device, limit_batches=10, limit_images=80):
    #Create predictions directory.
    os.makedirs('./visuals/predictions', exist_ok=True)

    model.eval()
    count = 0

    with torch.no_grad():
        for batch_idx, (batch_img, batch_mask) in enumerate(val_loader):
            if batch_idx >= limit_batches:
                break

            batch_img = batch_img.to(device)
            batch_mask = batch_mask.to(device)

            #Predict.
            pred_all = model(batch_img)
            pred_all = F.softmax(pred_all, dim=1)

            #Move to CPU and convert to numpy.
            batch_img = batch_img.cpu().numpy()
            batch_mask = batch_mask.cpu().numpy()
            pred_all = pred_all.cpu().numpy()

            for j in range(batch_img.shape[0]):
                count += 1

                #Denormalize image for display.
                img_display = batch_img[j].transpose(1, 2, 0)
                img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_display = np.clip(img_display, 0, 1)

                #Convert masks to RGB.
                true_mask_rgb = onehot_to_rgb(batch_mask[j].transpose(1, 2, 0), id2code)
                pred_mask_rgb = onehot_to_rgb(pred_all[j].transpose(1, 2, 0), id2code)

                #Plot trio and save.
                fig = plt.figure(figsize=(20, 8))

                ax1 = fig.add_subplot(1, 3, 1)
                ax1.imshow(img_display)
                ax1.set_title('Original Image')
                ax1.grid(False)
                ax1.axis('off')

                ax2 = fig.add_subplot(1, 3, 2)
                ax2.imshow(true_mask_rgb)
                ax2.set_title('Ground Truth Mask')
                ax2.grid(False)
                ax2.axis('off')

                ax3 = fig.add_subplot(1, 3, 3)
                ax3.imshow(pred_mask_rgb)
                ax3.set_title('Predicted Mask')
                ax3.grid(False)
                ax3.axis('off')

                plt.savefig(f'./visuals/predictions/prediction_{count}.png', facecolor='w', transparent=False, bbox_inches='tight', dpi=200)
                plt.close()

                if count >= limit_images:
                    break

            if count >= limit_images:
                break

    print(f"Generated {count} predictions in './visuals/predictions/' directory")


#This program performs semantic segmentation on drone imagery using a VGG16-based U-Net architecture with PyTorch.
def main():
    os.makedirs('./visuals', exist_ok=True)
    device = setup_device_and_safety()

    #Optional: demo augmentation visualization if sample exists.
    image_path = "./semantic_drone_dataset/training_set/images/040.jpg"
    mask_path = "./semantic_drone_dataset/training_set/gt/semantic/label_images/040.png"
    if os.path.exists(image_path) and os.path.exists(mask_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        transformed = transform(image=image, mask=mask)
        visualize(transformed['image'], transformed['mask'], image, mask)

    #Load class info.
    label_names, label_codes, code2id, id2code, name2id, id2name = load_class_dict()

    #Data loaders.
    train_loader, val_loader = prepare_dataloaders(id2code, device)

    #Model and training components.
    model, criterion, optimizer, scheduler = init_model_and_optim(device)

    #Train.
    train_losses, train_dices, val_losses, val_dices, learning_rates = train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        num_epochs=20, patience=6
    )

    #Save history and plots.
    save_history_and_plot(train_losses, train_dices, val_losses, val_dices, learning_rates)

    #Load best and generate predictions.
    checkpoint = torch.load('models/vgg16_unet_best.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Best model loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"Val Loss: {checkpoint['val_loss']:.4f}, Val Dice: {checkpoint['val_dice']:.4f}")

    generate_predictions_and_samples(model, val_loader, id2code, device)
    print(f"\nGenerated 80 predictions in './visuals/predictions/' directory")

    #Final summary.
    print("\nTRAINING SUMMARY")
    print(f"Framework: PyTorch {torch.__version__}")
    print(f"Device: {device}")
    print(f"Total epochs: {len(train_losses)}")
    print(f"Best validation loss: {min(val_losses):.4f}")
    print(f"Best validation dice: {max(val_dices):.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final training dice: {train_dices[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Final validation dice: {val_dices[-1]:.4f}")
    print("\nModel saved at: models/vgg16_unet_best.pth")
    print("Training history saved at: model_training_csv.log")
    print("Predictions saved in: ./visuals/predictions/")
    print("Training metrics plot saved at: ./visuals/model_metrics_plot.png")
    print("Repository: https://github.com/WhiteMetagross/SemanticSegmentationDroneImagesPyTorch")


if __name__ == "__main__":
    main()
