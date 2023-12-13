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
from util.fundus_prep import imread, imwrite, process_without_gb
import traceback
import toml

# Load configurations from toml file
with open("test_state.toml", "r") as toml_file:
    test_config = toml.load(toml_file)

# Access your variables
modelStore = test_config["test"]["modelStore"]
model_folder = test_config["test"]["model_folder"]
use_thresholding = test_config["test"]["use_thresholding"]
imagedir = test_config["test"]["imagedir"]

if not os.path.exists(modelStore):
    os.makedirs(modelStore)

model_path = os.path.join(modelStore, model_folder)
predictions_path = os.path.join(model_path, 'predictions')
if not os.path.exists(predictions_path):
    os.makedirs(predictions_path)


#TODO: come back to this and clean up when you decide on a final config!

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
except:
    print('No training configuration found.')
try:
    input_size = training_config['input_size']
    try:
        remove_background = training_config['rmbg']
    except:
        remove_background = training_config['remove_background']
    try:
        drop_path = training_config['drop_path']
    except:
        drop_path = training_config['drop_path_rate']
    weight_decay = training_config['weight_decay']
    layer_decay = training_config['layer_decay']
except:
    traceback.print_exc()
    print('No training configuration found. Using default input size.')
    input_size = 224
    remove_background = False
    drop_path = 0.2
    weight_decay = 0.0
    layer_decay = 0.0



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
def generate_visualization(transformed_image, class_index=None, model=None, use_thresholding=False, patch_size=16):
    # Generate LRP
    transformer_attribution = attribution_generator.generate_LRP(transformed_image,
                                                                 method="transformer_attribution",
                                                                 index=class_index).detach()

    # Move to CPU and convert to numpy
    transformer_attribution = transformer_attribution.cpu().numpy()

    # Calculate the grid size based on patch size
    input_height, input_width = transformed_image.shape[2], transformed_image.shape[3]  # Assuming transformed_image is 4D
    grid_size = (input_height // patch_size, input_width // patch_size)

    # Reshape and interpolate
    transformer_attribution = transformer_attribution.reshape(1, 1, *grid_size)
    transformer_attribution = torch.nn.functional.interpolate(torch.tensor(transformer_attribution),
                                                              size=(input_height, input_width),
                                                              mode='bilinear', align_corners=True)
    transformer_attribution = transformer_attribution.reshape(input_height, input_width).numpy()

    # Normalize transformer attribution
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    # Threshold the transformer attribution
    if use_thresholding:
        transformer_attribution = transformer_attribution * 255
        transformer_attribution = transformer_attribution.astype(np.uint8)
        ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255,
                                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        transformer_attribution[transformer_attribution == 255] = 1



    # Convert the original PIL Image to a NumPy array for processing

    # Convert the original PIL Image to a NumPy array for processing
    original_image_np = np.array(transformed_image.cpu())
    image_transformer_attribution = original_image_np.squeeze(0).transpose(1, 2,
                                                                           0)  # Also ensure this is correctly reshaped
    # Create heatmap from mask on image
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    return vis


# Before inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model with the correct number of classes
model = models_vit.__dict__['vit_large_patch16'](num_classes=num_classes, drop_path_rate=drop_path, global_pool=True,img_size=input_size)

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

# Define your image transforms
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    # Add any other transforms used during training
])
if remove_background == True:
    transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.ToTensor()(Image.fromarray(process_without_gb(np.array(img), np.array(img), [], [], [])[0]))),
        transforms.Resize((input_size, input_size)),
    ])

# Load and preprocess the image
imagedir = imagedir
images = [f for f in os.listdir(imagedir) if os.path.isfile(os.path.join(imagedir, f))]


# pick 3 random images from the list of images in the folder
def random_images(images):
    random_images = []
    for i in range(3):
        random_images.append(images[random.randint(0, len(images) - 1)])
    return random_images


selected_images = random_images(images)
print(selected_images)
count = 0

# Initialize explainability modules
model_explain = vit_LRP(pretrained=True, checkpoint=weightpath,img_size = input_size)  # Ensure this returns the LRP-capable model
model_explain = model_explain.to(device)
model_explain.eval()

print("Model loaded for explainability")
attribution_generator = LRP(model_explain)
print("Attribution generator loaded")
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
        prediction = output.argmax(dim=1)
        print(prediction)

    # Print the prediction
    print("Predicted class:", prediction.item())
    print("Predicted class name:", classes[prediction.item()])
    # display the % confidence
    print("Confidence:", torch.nn.functional.softmax(output, dim=1)[0][prediction.item()].item())

    # display the class and % confidence for all classes in a nice format
    print("All classes and confidence:")
    # sort by confidence
    sorted, indices = torch.sort(torch.nn.functional.softmax(output, dim=1)[0], descending=True)
    for i in range(num_classes):
        print(classes[indices[i]], sorted[i].item())

    original_image = Image.open(image_path)
    print("Image loaded")
    top_class_name = classes[top_prediction]
    confidence = probabilities[top_prediction].item()
    print("Top class name:", top_class_name)
    original_image.show()

    original_image.save(os.path.join(predictions_path, f'{s}-{count}-original_image-{top_class_name}-{confidence:4f}.jpg'))

    # Display top predictions with confidence
    print("Top Predictions:")
    for i in range(top_predictions.indices.size(0)):
        class_index = top_predictions.indices[i].item()
        confidence = top_predictions.values[i].item()
        print(f"Class: {classes[class_index]}, Confidence: {confidence:.4f}")
        class_name = classes[class_index]
        print("Class name:", class_name)

        vis = generate_visualization(image, class_index=class_index)
        vis_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        vis_image.show()  # or vis_image.save(f'output_class_{class_index}.jpg')
        vis_image.save(os.path.join(predictions_path, f'{s}-{count}-vis_image-{class_name}-{confidence:.4f}.jpg'))
        if use_thresholding == True:
            vis = generate_visualization(image, class_index=class_index, use_thresholding=True)
            vis_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            vis_image.show()
            vis_image.save(os.path.join(predictions_path, f'{s}-{count}-vis_image-{class_name}-{confidence:.4f}-thresholded.jpg'))

        count += 1
