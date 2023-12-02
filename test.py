import torch
import models_vit
import json
from PIL import Image
import torchvision.transforms as transforms
import os
from baselines.ViT.ViT_LRP import vit_large_patch16_224_ch as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

modelStore = './models/'
if not os.path.exists(modelStore):
    os.makedirs(modelStore)

model_folder = 'ch-tagstest-12-02-2023-024214'
model_path = os.path.join(modelStore, model_folder)

use_thresholding = False

# Paths to the configuration files
dataset_info_path = os.path.join(model_path, 'dataset_info.json')
training_config_path = os.path.join(model_path, 'training_config.json')

# Path to the model weights
weightpath = os.path.join(model_path, 'checkpoint-best.pth')

try:
    # Read dataset information
    with open(dataset_info_path, 'r') as file:
        dataset_info = json.load(file)
    num_classes = dataset_info['num_classes']
    classes = dataset_info['classes']
except:
    print('No dataset information found. Using default number of classes.')
    num_classes = 2

try:
    # Read training configuration (if necessary)
    with open(training_config_path, 'r') as file:
        training_config = json.load(file)
    input_size = training_config['input_size']
except:
    print('No training configuration found. Using default input size.')
    input_size = 224


def show_cam_on_image(img, mask):
    print("Mask shape:", mask.shape)
    print("Image shape:", img.shape)

    # If img is a NumPy array and has a batch dimension, reshape it
    if img.ndim == 4:  # Check if it's a 4D array
        img = img.squeeze(0)  # Remove batch dimension
        img = img.transpose(1, 2, 0)  # Rearrange dimensions to [height, width, channels]
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to range [0, 1]

    # Generate heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Resize the image to match the heatmap dimensions
    resized_img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    cam = heatmap + np.float32(resized_img)
    cam = cam / np.max(cam)
    return cam


# Function to generate visualization
def generate_visualization(transformed_image, class_index=None, model=None):
    # Ensure the image tensor is on the correct device

    # Generate LRP
    transformer_attribution = attribution_generator.generate_LRP(transformed_image,
                                                                 method="transformer_attribution",
                                                                 index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16,
                                                              mode='bilinear',
                                                              align_corners=True)
    transformer_attribution = transformer_attribution.reshape(224,
                                                              224).data.cpu().numpy()  # Move to CPU before converting to NumPy
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())

    if use_thresholding:
        transformer_attribution = transformer_attribution * 255
        transformer_attribution = transformer_attribution.astype(np.uint8)
        ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255,
                                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        transformer_attribution[transformer_attribution == 255] = 1

    # Convert the original PIL Image to a NumPy array for processing

    original_image_np = np.array(transformed_image.cpu())
    image_transformer_attribution = original_image_np / 255.0  # Normalize to range [0, 1]

    # Normalize transformer attribution
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())

    # Create heatmap from mask on image
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    return vis


# Before inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model with the correct number of classes
model = models_vit.__dict__['vit_large_patch16'](num_classes=num_classes, drop_path_rate=0.2, global_pool=True)

# Load the state dictionary
checkpoint = torch.load(weightpath, map_location=device)
checkpoint_model = checkpoint['model']

# Remove keys that don't match
for k in ['head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias']:
    checkpoint_model.pop(k, None)

# Load the state dictionary into the model
model.load_state_dict(checkpoint_model, strict=False)
model.eval()
model.to(device)

# Initialize explainability modules
model_explain = vit_LRP(pretrained=True, checkpoint=weightpath)  # Ensure this returns the LRP-capable model
model_explain = model_explain.to(device)
model_explain.eval()

# Define your image transforms
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    # Add any other transforms used during training
])


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
count = 0
for s in range(len(selected_images)):
    image_path = os.path.join(imagedir, selected_images[s])
    image = Image.open(os.path.join(imagedir, selected_images[s]))
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Run inference
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        top_predictions = torch.topk(probabilities, min(5, num_classes))  # Get top predictions, up to 5
        top_prediction = top_predictions.indices[0].item()
        # Generate and display visualizations for each of the top predictions
    # Load and preprocess your test image
    original_image = Image.open(image_path)
    print("Image loaded")
    original_image.show()
    top_class_name = classes[top_prediction]
    original_image.save(os.path.join(model_path, f'{s}-{count}-original_image-{top_class_name}.jpg'))

    # Display top predictions with confidence
    print("Top Predictions:")
    for i in range(top_predictions.indices.size(0)):
        class_index = top_predictions.indices[i].item()
        confidence = top_predictions.values[i].item()
        print(f"Class: {classes[class_index]}, Confidence: {confidence:.4f}")
        class_name = classes[class_index]
        print("Class name:", class_name)

        attribution_generator = LRP(model_explain)
        print("Attribution generator loaded")




        print("Generating visualization")


        vis = generate_visualization(image, class_index=class_index)
        vis_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        # Display or save the result
        vis_image.show()  # or vis_image.save(f'output_class_{class_index}.jpg')
        vis_image.save(os.path.join(model_path, f'{s}-{count}-vis_image-{class_name}.jpg'))