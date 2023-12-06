import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models_vit import ViTForImageReconstruction
import models_vit
import os
import random

# Initialize the model (make sure this matches the initialization from your training script)
base_vit_model = models_vit.vit_large_patch16
model = ViTForImageReconstruction(base_vit_model, decoder_embed_dim=512, drop_path_rate=0.2, global_pool=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the trained model weights
model.load_state_dict(torch.load('./models/Leaky-12-05-2023-021340/checkpoint-best.pth', map_location=device))

# Set the model to evaluation mode
model.eval()

# Prepare the input image (make sure to replace 'path_to_input_image' with your actual file path)
# Load and preprocess the image
imagedir = 'D:\\data\\RetFound\\images'
images = [f for f in os.listdir(imagedir) if os.path.isfile(os.path.join(imagedir, f))]
# create function to pick 3 random images from the list of images in the folder
def random_images(images):
    random_images = []
    for i in range(3):
        random_images.append(images[random.randint(0, len(images) - 1)])
    return random_images
selected_images = random_images(images)
print(selected_images)
for s in range(len(selected_images)):
    input_image = Image.open(os.path.join(imagedir,selected_images[s])).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    # Reconstruct the image
    with torch.no_grad():  # No need to track gradients for visualization
        reconstructed_img = model(input_tensor)

    # Convert the reconstructed image tensor to a PIL image
    reconstructed_img = reconstructed_img.squeeze(0).cpu()  # Remove batch dimension and move to cpu


    # Check min and max values
    print("Min value:", reconstructed_img.min().item())
    print("Max value:", reconstructed_img.max().item())


    non_zero_count = (reconstructed_img > 0).sum()
    print("Number of non-zero pixels:", non_zero_count.item())

    # Scale the pixel values to [0, 1]
    min_val = reconstructed_img.min()
    range_val = reconstructed_img.max() - min_val
    if range_val > 0:
        # Avoid division by zero
        reconstructed_img = (reconstructed_img - min_val) / range_val




    # If using a sigmoid in the last layer of the decoder, this should not be necessary

    reconstructed_img = transforms.ToPILImage()(reconstructed_img)

    # Display the original and reconstructed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(input_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Reconstructed Image')
    plt.imshow(reconstructed_img)
    plt.axis('off')

    plt.show()
