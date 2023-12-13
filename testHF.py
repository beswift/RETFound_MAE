from PIL import Image
import os
import torch
from transformers import ViTFeatureExtractor, AutoModelForImageClassification
import random
import toml

# Load configurations from toml file
with open("test_state.toml", "r") as toml_file:
    test_config = toml.load(toml_file)

# Access your variables
imagedir = test_config["test"]["imagedir"]

# Load model and processor
model = AutoModelForImageClassification.from_pretrained("bswift/test")
processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224",size=256)
# Update the keys to match the expected format in VisionTransformerForImageClassification
updated_state_dict = {f'vit.{k}' if not k.startswith('vit.') else k: v for k, v in model.state_dict().items()}

# Load the updated state dict into your model
model.load_state_dict(updated_state_dict, strict=False)
model.eval()


# Load and preprocess the image
images = [f for f in os.listdir(imagedir) if os.path.isfile(os.path.join(imagedir, f))]

selected_images = random.sample(images, min(len(images), 3))
print("Selected images:", selected_images)

for image_name in selected_images:
    image_path = os.path.join(imagedir, image_name)
    image = Image.open(image_path)
    image.show()

    # Preprocess the image
    inputs = processor(image, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Process outputs
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top_results = torch.topk(probabilities, 1).indices[0]

    for idx in top_results:
        confidence = probabilities[0, idx.item()].item()
        print(f"Class {idx.item()} Confidence: {confidence:.4f}")
