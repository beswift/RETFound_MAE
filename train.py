import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import os
from util.data_handler import split_dataset, check_images
import subprocess
import datetime
import json
import toml

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device}')

try:
    # Load configurations from toml file
    with open("train_state.toml", "r") as toml_file:
        config = toml.load(toml_file)
except:
    print('Error loading toml file')
    if not os.path.exists('train_state.toml'):
        print('File does not exist')
        os.makedirs('train_state.toml')
    else:
        print('File exists but could not be loaded')

# Now access your variables like this
cfpweightpath = config["training"]["cfpweightpath"]
octweightpath = config["training"]["octweightpath"]
parent_folder = config["training"]["parent_folder"]
print(f'Parent folder: {parent_folder}')
output_folder = config["training"]["output_folder"]
batch_size = config["training"]["batch_size"]
world_size = config["training"]["world_size"]
epochs = config["training"]["epochs"]
base_model = config["training"]["base_model"]
ft_weightpath = config["training"]["ft_weightpath"]
blr = config["training"]["blr"]
layer_decay = config["training"]["layer_decay"]
weight_decay = config["training"]["weight_decay"]
drop_path = config["training"]["drop_path"]
num_classes = config["training"]["num_classes"]
task = config["training"]["task"]
rmbg = config["training"]["rmbg"]
input_size = config["training"]["input_size"]
use_cases = config["training"]["use_cases"]
limitations = config["training"]["limitations"]
ethics = config["training"]["ethics"]
authors = config["training"]["authors"]
references = config["training"]["references"]
intended_use = config["training"]["intended_use"]

# we'll use the time in the output folder name to avoid overwriting previous results - so clean it up
time = datetime.datetime.now().strftime("%m-%d-%Y-%H%M%S").replace(' ', '_').replace(':', '')

# create output folder if it doesn't exist
if output_folder == '':
    output_folder = os.path.join(parent_folder, 'outputs', f'_artifacts_{time}')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# if parent folder is not split (i.e does not contain train, val, test folders), split it
dirs = os.listdir(parent_folder)
print(f'Parent folder contains: {dirs}')
if 'train' not in dirs or 'val' not in dirs or 'test' not in dirs:
    print(f'Parent folder {parent_folder} is not split. Splitting now.')
    data_folder = split_dataset(parent_folder, remove_background=rmbg)
else:
    data_folder = parent_folder

if task == '':
    task = os.path.join(output_folder, f'task-{time}')
    model_folder_name = f'task-{time}'
else:
    task = f'./models/{task}-{time}/'
    model_folder_name = f'{task}-{time}'

try:
    if not os.path.exists(task):
        os.makedirs(task)
except:
    print(f'Error creating task folder: {task}')

check_images(data_folder)

num_classes = len(
    [d for d in os.listdir(os.path.join(data_folder, 'train')) if os.path.isdir(os.path.join(data_folder, 'train', d))])
# number of training images is the total number of files in all subfolders of train
num_training_images = sum([len(files) for r, d, files in os.walk(os.path.join(data_folder, 'train'))])
# classes are the subfolders of train only (i.e. the class names)
classes = [d for d in os.listdir(os.path.join(data_folder, 'train')) if
           os.path.isdir(os.path.join(data_folder, 'train', d))]

print(f'Number of classes: {num_classes}')
print(f'Classes: {classes}')
print(f'Number of training images: {num_training_images}')

# call the model
model = models_vit.__dict__[base_model](
    num_classes=num_classes,
    drop_path_rate=drop_path,
    global_pool=True,
    img_size=input_size,
)

checkpoint = torch.load(ft_weightpath, map_location=device)

checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# interpolate position embedding
interpolate_pos_embed(model, checkpoint_model)

# load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=False)

assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

# manually initialize fc layer
trunc_normal_(model.head.weight, std=2e-5)

print("Model = %s" % str(model))

# Save dataset information
dataset_info = {
    'num_classes': num_classes,
    'classes': classes,
    'num_training_images': num_training_images
}

dataset_info_path = os.path.join(task, 'dataset_info.json')
with open(dataset_info_path, 'w') as f:
    json.dump(dataset_info, f)

# Save training configuration
training_config = {
    'batch_size': batch_size,
    'epochs': epochs,
    'model': 'vit_large_patch16',
    'base_learning_rate': blr,
    'layer_decay': layer_decay,
    'weight_decay': weight_decay,
    'drop_path_rate': drop_path,
    'input_size': input_size,
    'num_classes': num_classes,
    'task': task,
    'output_dir': output_folder,
    'world_size': world_size,
    'finetune': ft_weightpath,
    'rmbg': rmbg,
    # Add any other relevant configuration parameters here
}

training_config_path = os.path.join(task, 'training_config.json')
with open(training_config_path, 'w') as f:
    json.dump(training_config, f)

# Prepare command for fine-tuning
command = [
    'python', 'main_finetune.py',
    '--data_path', data_folder,  # Use the output folder from split_dataset
    '--batch_size', str(batch_size),
    '--world_size', str(world_size),
    '--epochs', str(epochs),
    '--model', base_model,
    '--finetune', ft_weightpath,
    '--blr', str(blr),
    '--layer_decay', str(layer_decay),
    '--weight_decay', str(weight_decay),
    '--drop_path', str(drop_path),
    '--nb_classes', str(num_classes),
    '--task', f'./{task}/',
    '--output_dir', output_folder,
    '--input_size', str(input_size),

    # Add other necessary arguments here
]

# Run the fine-tuning command
subprocess.run(command)

# Config for the model
config = {
    "model_type": "ViT",
    "architecture": base_model,
    "input_size": input_size,
    "num_classes": num_classes,
    "drop_path_rate": drop_path,
    "layer_decay": layer_decay,
    "weight_decay": weight_decay,
    "base_learning_rate": blr,
    "batch_size": batch_size,
    "epochs": epochs,
    "remove_background": rmbg,
}

# Content for the model card
model_card = {
    "model_name": f'{task}-{time}',
    "model_type": "ViT",
    "description": f'Fine-tuned {base_model} model for {task}',
    "use_cases": use_cases,
    "limitations": limitations,
    "ethics": ethics,
    "training_data": f'{num_training_images} images from {num_classes} classes',
    "training_procedure": f'Fine-tuned for {epochs} epochs with batch size {batch_size} and base learning rate {blr}',
    "intended_use": intended_use,
    "authors": authors,
    "references": references,
}

# Save config.json
config_path = os.path.join(task, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config, f)

# Save model_card.json
model_card_path = os.path.join(task, 'model_card.json')
with open(model_card_path, 'w') as f:
    json.dump(model_card, f)

# Convert model_card to Markdown format and save as README.md
model_card_md = f"""# {model_card["model_name"]}
## Description
{model_card["description"]}

## Use Cases
- {model_card["use_cases"][0]}
- {model_card["use_cases"][1]}

## Limitations
- {model_card["limitations"][0]}
- {model_card["limitations"][1]}

## Ethics
- {model_card["ethics"][0]}
- {model_card["ethics"][1]}

## Training Data
{model_card["training_data"]}

## Training Procedure
{model_card["training_procedure"]}

## Intended Use
{model_card["intended_use"]}

## Authors
- {model_card["authors"][0]}
- {model_card["authors"][1]}

## References
- {model_card["references"][0]}
- {model_card["references"][1]}
"""

readme_path = os.path.join(task, 'README.md')
with open(readme_path, 'w') as f:
    f.write(model_card_md)

# Create a requirements.txt file for huggingface
requirements = [
    "torch==1.8.1+cu111",
    "timm==0.3.2",
    "torchvision==0.9.1+cu111",
    "torchaudio==0.8.1",
    "opencv-python>=4.5.3.56",
    "pandas>=0.25.3",
    "Pillow>=8.3.1",
    "protobuf>=3.17.3",
    "pycm>=3.2",
    "pydicom>=2.3.0",
    "scikit-image>=0.17.2",
    "scikit-learn>=0.24.2",
    "scipy>=1.5.4",
    "tensorboard>=2.6.0",
    "tensorboard-data-server>=0.6.1",
    "tensorboard-plugin-wit>=1.8.0",
    "tqdm>=4.62.1",
    "einops>=0.3.0",
    "h5py>=2.8.0",
    "imageio>=2.9.0",
    "matplotlib>=3.3.2",
    "tqdm>=4.51.0",
    "transformers>=3.5.1",
    "utils>=1.0.1",
    "Pygments>=2.7.4",
    "pytorch-msssim>=1.0.0",
    "toml",
]
requirements_path = os.path.join(task, 'requirements.txt')
with open(requirements_path, 'w') as f:
    f.writelines(f"{req}\n" for req in requirements)

test_toml = toml.load('test_state.toml')
# update model_folder to the new model folder
test_toml["test"]["model_folder"] = model_folder_name
test_toml["test"]["input_size"] = input_size

with open('test_state.toml', 'w') as toml_file:
    toml.dump(test_toml, toml_file)
