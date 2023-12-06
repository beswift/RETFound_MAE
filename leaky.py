import torch
import models_vit
import os
import datetime
import json
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models_vit import ViTForImageReconstruction, CustomLoss,CombinedLoss
from util.pos_embed import interpolate_pos_embed
from util.data_handler import log_sample_reconstruction_to_tensorboard, display_sample_reconstruction, PairedImageDataset

# When initializing the loss function in your training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



fundus_folder = os.path.join('Data','Leaky','Fundus')
fa_folder = os.path.join('Data','Leaky','FA')

dataset_name = 'Leaky'
batch_size = 10
epochs = 100
blr = 5e-3
weight_decay = 0.05
num_folds = 2
loss_function = CombinedLoss(device)
kfold = KFold(n_splits=num_folds, shuffle=True)


time = datetime.datetime.now().strftime("%m-%d-%Y-%H%M%S").replace(' ', '_').replace(':', '')

output_folder = os.path.join('output_dir','models',f'{dataset_name}-{time}')
print(output_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model_folder = os.path.join('models',f'{dataset_name}-{time}')
print(model_folder)
if not os.path.exists(model_folder):
    os.makedirs(model_folder)


output_folder = os.path.join('output_dir','models',f'leaky-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set up TensorBoard
writer = SummaryWriter(os.path.join(output_folder))  # Adjust path as needed


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


train_dataset = PairedImageDataset(fundus_folder, fa_folder, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define model type and other parameters
base_model = 'vit_large_patch16'  # Example model type
#num_classes = 2  # Example for classification, adjust according to your task


# Define the base Vision Transformer model
base_vit_model = models_vit.vit_large_patch16

# Initialize the model with the new class
model = ViTForImageReconstruction(base_vit_model, decoder_embed_dim=512,  drop_path_rate=0.2, global_pool=True)

#check for gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load RETFound weights for color fundus photography
cfpweightpath = os.path.join('D:\\data\\RetFound\\weights', 'RETFound_cfp_weights.pth')

ft_weightpath = cfpweightpath

# Load pre-trained weights for the encoder
checkpoint = torch.load(ft_weightpath, map_location=device)

# Extract encoder weights (assuming they are stored in a dict under 'model')
encoder_weights = {k: v for k, v in checkpoint['model'].items() if 'decoder' not in k}

# Load the encoder weights
model.load_state_dict(encoder_weights, strict=False)

# Initialize decoder weights
# Apply your custom weight initialization for decoder layers
# For example, using xavier_uniform initialization
def init_weights(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.decoder.apply(init_weights)

interpolate_pos_embed(model.vit, checkpoint['model'])


# Load weights into model, excluding incompatible layers
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict, strict=False)

# Initialize the model head if you have changed the task or the number of classes
#trunc_normal_(model.head.weight, std=0.02)
#trunc_normal_(model.vit.head.weight, std=0.02)


# Store the average performance across folds
average_performance = []

for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Define data loaders for training and validation
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_subsampler)
    val_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=val_subsampler)

    # Initialize the model for this fold
    model = ViTForImageReconstruction(base_vit_model, decoder_embed_dim=512, drop_path_rate=0.2, global_pool=True)
    model.to(device)  # Move the model to the appropriate device

    optimizer = torch.optim.AdamW(model.parameters(), lr=blr, weight_decay=weight_decay)
    criterion = loss_function
    # Usage
    #criterion = CustomLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=3, verbose=True)

    best_loss = float('inf')
    early_stopping_patience = 10
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (fundus_imgs, fa_imgs) in enumerate(train_loader):
            try:
                fundus_imgs, fa_imgs = fundus_imgs.to(device), fa_imgs.to(device)

                optimizer.zero_grad()

                # Forward pass
                reconstructed_imgs = model(fundus_imgs)

                # Calculate loss
                loss_value = criterion(reconstructed_imgs, fa_imgs)
                loss_value.backward()
                optimizer.step()

                running_loss += loss_value.item()

                # Log batch loss to TensorBoard
                writer.add_scalar('Loss/train', loss_value.item(), epoch * len(train_loader) + i)

            except Exception as e:
                print(f"Exception occurred during training: {e}")
                # Save model and writer state before exiting
                torch.save(model.state_dict(), os.path.join(output_folder, f'leaky-EXCEPTION-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.pth'))
                writer.close()
                raise e

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

        # Log epoch loss to TensorBoard
        writer.add_scalar('Loss/epoch', epoch_loss, epoch)
        scheduler.step(epoch_loss)

        # Save model checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), os.path.join(model_folder, 'checkpoint-best.pth'))
            display_sample_reconstruction(model, train_dataset, device)

            log_sample_reconstruction_to_tensorboard(writer, model, train_dataset, device, epoch, tag='Reconstruction/Best')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        print(f'Best Loss: {best_loss:.4f}')

        # Evaluate after training
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (val_fundus_imgs, val_fa_imgs) in enumerate(val_loader):
                val_fundus_imgs, val_fa_imgs = val_fundus_imgs.to(device), val_fa_imgs.to(device)
                val_reconstructed_imgs = model(val_fundus_imgs)
                loss = criterion(val_reconstructed_imgs, val_fa_imgs)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        average_performance.append(val_loss)

        print(f'Validation Loss for fold {fold}: {val_loss:.4f}\n')

    # Calculate and print the average performance across folds
    print(f'Average Validation Loss across folds: {np.mean(average_performance):.4f}')

    writer.close()



# Define the dataset information
dataset_info = {
    'number_of_images': len(train_dataset),
    'classes': ['fundus', 'FA'],  # Update as per your dataset classes
    'image_dimensions': (224, 224),
    'fundus_folder': fundus_folder,
    'fa_folder': fa_folder
}

# Write the dataset information to a JSON file
with open(os.path.join(model_folder, 'dataset_info.json'), 'w') as f:
    json.dump(dataset_info, f, indent=4)


# Define the training configuration
training_config = {
    'epochs': epochs,
    'batch_size': batch_size,
    'learning_rate': blr,
    'weight_decay': weight_decay,
    'input_size': (224, 224),  # Update if different
    'base_model': 'vit_large_patch16',
    'num_folds': num_folds,
    'loss_function': str(loss_function),
    'loss': str(loss),
}

# Write the training configuration to a JSON file
with open(os.path.join(model_folder, 'training_config.json'), 'w') as f:
    json.dump(training_config, f, indent=4)