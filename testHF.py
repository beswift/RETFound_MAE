from io import BytesIO

from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

def perform_test_inference(model_name, image_url):
    """
    Performs a test inference using the model from Hugging Face Hub.

    Args:
    model_name (str): Full path of the model on Hugging Face Hub (e.g., 'username/model_name').
    image_url (str): URL of an image to test the model.
    """
    # Load the model and feature extractor
    model = ViTForImageClassification.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    # Load and preprocess the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Perform inference
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    print(f"Predicted class index: {predicted_class_idx}")

# Example usage
test_model_name = "your-username/your-model-name"  # Change to your model's Hugging Face path
test_image_url = "https://example.com/test_image.jpg"  # URL of a test image

# Uncomment the following line to perform a test inference
# perform_test_inference(test_model_name, test_image_url)
