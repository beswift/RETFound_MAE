import os
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import traceback
import datetime
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from .fundus_prep import process_without_gb,imread,imwrite





class PairedImageDataset(Dataset):
    def __init__(self, fundus_folder, fa_folder, transform=None):
        self.fundus_folder = fundus_folder
        self.fa_folder = fa_folder
        self.transform = transform

        self.fundus_images = sorted(os.listdir(fundus_folder))
        self.fa_images = sorted(os.listdir(fa_folder))

    def __len__(self):
        return len(self.fundus_images)

    def __getitem__(self, idx):
        fundus_img_path = os.path.join(self.fundus_folder, self.fundus_images[idx])
        fa_img_path = os.path.join(self.fa_folder, self.fa_images[idx])

        fundus_img = Image.open(fundus_img_path).convert('RGB')
        fa_img = Image.open(fa_img_path).convert('L')  # 'L' mode for grayscale

        if self.transform:
            fundus_img = self.transform(fundus_img)
            fa_img = self.transform(fa_img)

        return fundus_img, fa_img


# Function to display a sample image and its reconstruction
def display_sample_reconstruction(model, dataset, device):
    model.eval()
    with torch.no_grad():
        # Randomly choose an image from the dataset
        idx = random.randint(0, len(dataset) - 1)
        original_img, _ = dataset[idx]
        original_img = original_img.unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Reconstruct the image
        reconstructed_img = model(original_img).squeeze(0).cpu()

        # Post-process the reconstructed image
        # (Add any necessary post-processing steps here)

        # Convert tensors to PIL images for display
        original_img_pil = transforms.ToPILImage()(original_img.squeeze(0))
        reconstructed_img_pil = transforms.ToPILImage()(reconstructed_img)

        # Display the images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img_pil)
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_img_pil)
        plt.title("Reconstructed Image")
        plt.axis("off")
        # show the plot for 2 seconds and then save it
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def log_sample_reconstruction_to_tensorboard(writer, model, dataset, device, epoch, tag='Reconstruction'):
    model.eval()
    with torch.no_grad():
        # Randomly choose an image from the dataset
        idx = random.randint(0, len(dataset) - 1)
        original_img, _ = dataset[idx]
        original_img_tensor = original_img.unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Reconstruct the image
        reconstructed_img = model(original_img_tensor).squeeze(0)  # Remove batch dimension

        # Convert grayscale to RGB by replicating the single channel
        reconstructed_img_rgb = reconstructed_img.repeat(3, 1, 1)  # Repeat channel 3 times

        # Create an image grid
        img_grid = torch.cat((original_img_tensor.squeeze(0), reconstructed_img_rgb), dim=2)  # Concatenate along width
        writer.add_images(tag, img_grid.unsqueeze(0), epoch)  # Add batch dimension for TensorBoard


def split_dataset(parent_folder, train_size=0.7, val_size=0.15, test_size=0.15, remove_background=False):

    #exclude dirs that start with dataset from classes
    classes = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f)) and not f.startswith('_dataset')]
    time = datetime.datetime.now().strftime("%m-%d-%Y-%H%M%S")

    output_folder = os.path.join(parent_folder,'_datasets', f'_dataset_{time}', 'training_data')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    skipped_classes = []

    for cls in classes:
        cls_path = os.path.join(parent_folder, cls)
        images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        try:
            train_val, test = train_test_split(images, test_size=test_size)
            train, val = train_test_split(train_val, test_size=val_size/(train_size + val_size))
        except:
            print(f'Error splitting {cls} - most likely not enough images')
            skipped_classes.append(cls)
            traceback.print_exc()
            continue

        if not os.path.exists(os.path.join(output_folder, 'train', cls)):
            os.makedirs(os.path.join(output_folder, 'train', cls))
        if not os.path.exists(os.path.join(output_folder, 'val', cls)):
            os.makedirs(os.path.join(output_folder, 'val', cls))
        if not os.path.exists(os.path.join(output_folder, 'test', cls)):
            os.makedirs(os.path.join(output_folder, 'test', cls))

        try:

            for subset in ['train', 'val', 'test']:
                for img in locals()[subset]:
                    try:
                        img_path = os.path.join(cls_path, img)
                        output_path = os.path.join(output_folder, subset, cls, img)

                        if remove_background:
                            # Process the image to remove background
                            print(f'removing background from {img}')
                            image = imread(img_path)  # Ensure imread function is defined or imported
                            processed_img, borders, _, label,_,_,_ = process_without_gb(image, image, [], [], [])
                            imwrite(output_path, processed_img)  # Save the processed image
                            imwrite(output_path.replace('.jpg', '_label.jpg'), label)  # Save the label image
                            print(f'Processed {img}')
                        else:
                            # Copy the image without processing
                            shutil.copy(img_path, output_path)

                    except Exception as e:
                        print(f'Error processing {img}: {e}')
                        traceback.print_exc()
        except:
            print(f'Error splitting {cls}')
            traceback.print_exc()
            skipped_classes.append(cls)
            continue

    return output_folder


from PIL import Image
import os

def check_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    with Image.open(os.path.join(root, file)) as img:
                        img.verify()  # Verify if it's an image
                except (IOError, SyntaxError) as e:
                    print(f'Bad file:', file)  # Print out the names of corrupt files
