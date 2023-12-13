import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
import json
import random
import toml
from models_vit import vit_large_patch16, VisionTransformerForImageClassification # Import your custom model function

from transformers import ViTFeatureExtractor


# Load configurations from toml file
with open("test_state.toml", "r") as toml_file:
    test_config = toml.load(toml_file)

# Access your variables
modelStore = test_config["test"]["modelStore"]
model_folder = test_config["test"]["model_folder"]
imagedir = test_config["test"]["imagedir"]
input_size = test_config["test"]["input_size"]

model_directory = os.path.join(modelStore, model_folder)
print(model_directory)
print(os.listdir(model_directory))

# Paths to the configuration files
dataset_info_path = os.path.join(modelStore, model_folder, 'dataset_info.json')

try:
    # Read dataset information
    with open(dataset_info_path, 'r') as file:
        dataset_info = json.load(file)
    num_classes = dataset_info['num_classes']
    classes = dataset_info['classes']

except FileNotFoundError:
    print('No dataset information found. Using default number of classes.')
    num_classes = 2


from transformers import ViTConfig

# Load configuration from file
config_path = os.path.join(model_directory, 'config.json')
with open(config_path, 'r') as file:
    config_dict = json.load(file)
config = ViTConfig.from_dict(config_dict)

# Initialize the model with the loaded configuration
model = VisionTransformerForImageClassification(config, num_classes=num_classes, img_size=input_size)
# Load your trained model's state dict
model_path = os.path.join(model_directory, 'checkpoint-best.pth')
checkpoint = torch.load(model_path, map_location='cpu')

# Update the keys to match the expected format in VisionTransformerForImageClassification
updated_state_dict = {f'vit.{k}' if not k.startswith('vit.') else k: v for k, v in checkpoint['model'].items()}

# Load the updated state dict into your model
model.load_state_dict(updated_state_dict, strict=False)
model.eval()  # Set the model to evaluation mode
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224', size=input_size)



# Load and preprocess the image
images = [f for f in os.listdir(imagedir) if os.path.isfile(os.path.join(imagedir, f))]

def random_images(images, num=3):
    return random.sample(images, min(len(images), num))

selected_images = random_images(images)
print("Selected images:", selected_images)

for image_name in selected_images:
    image_path = os.path.join(imagedir, image_name)
    image = Image.open(image_path)
    image.show()

    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Process outputs (e.g., extract logits or probabilities)
    logits = outputs.logits  # Extracts the logits tensor from the outputs
    probabilities = torch.nn.functional.softmax(logits, dim=-1)  # Apply softmax to the logits tensor

    top_results = torch.topk(probabilities, num_classes).indices[0]

    # Print predictions
    print(f"Predictions for {image_name}:")
    for idx in top_results:
        class_idx = idx.item()
        confidence = probabilities[0, class_idx].item()
        print(f"  - {classes[class_idx]}: {confidence:.4f}")
