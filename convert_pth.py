import torch
import models_vit
import toml
import os

# Function to load the configuration file
def load_config(config_file):
    try:
        with open(config_file, "r") as toml_file:
            return toml.load(toml_file)
    except Exception as e:
        print(f'Error loading the TOML configuration file: {e}')
        raise

# Load configurations from the TOML file
config = load_config("train_state.toml")
test_config = load_config("test_state.toml")

checkpoint_store = test_config["test"]["modelStore"]
checkpoint_folder = test_config["test"]["model_folder"]
checkpoint_dir = os.path.join(checkpoint_store, checkpoint_folder)

# Extract relevant configuration parameters
base_model = config["training"]["base_model"]
input_size = config["training"]["input_size"]
num_classes = config["training"]["num_classes"]
drop_path = config["training"]["drop_path"]
ft_weightpath = checkpoint_dir + '/checkpoint-best.pth'

# Initialize the model
# Note: Add other parameters if your model initialization requires them
model = models_vit.__dict__[base_model](
    num_classes=num_classes,
    drop_path_rate=drop_path,
    global_pool=True,
    img_size=input_size
)

# Load the checkpoint
checkpoint = torch.load(ft_weightpath, map_location='cpu')
model.load_state_dict(checkpoint['model'])

# Save the model as pytorch_model.bin
torch_model_name = 'pytorch_model.bin'
torch_model_save_path = os.path.join(checkpoint_dir, torch_model_name)
torch.save(model.state_dict(), torch_model_save_path)
print("Model saved as pytorch_model.bin")
