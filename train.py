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
    output_folder = os.path.join(parent_folder,'outputs', f'_artifacts_{time}')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# if parent folder is not split (i.e does not contain train, val, test folders), split it
if not os.path.exists(os.path.join(parent_folder, 'train')):
    data_folder = split_dataset(parent_folder,remove_background=rmbg)
else:
    data_folder = parent_folder

if task == '':
    task = os.path.join(output_folder, f'task-{time}')
else:
    task = f'./models/{task}-{time}/'

try:
    if not os.path.exists(task):
        os.makedirs(task)
except:
    print(f'Error creating task folder: {task}')


check_images(data_folder)

num_classes = len([d for d in os.listdir(os.path.join(data_folder, 'train')) if os.path.isdir(os.path.join(data_folder, 'train', d))])
#number of training images is the total number of files in all subfolders of train
num_training_images = sum([len(files) for r, d, files in os.walk(os.path.join(data_folder, 'train'))])
#classes are the subfolders of train only (i.e. the class names)
classes = [d for d in os.listdir(os.path.join(data_folder, 'train')) if os.path.isdir(os.path.join(data_folder, 'train', d))]

print(f'Number of classes: {num_classes}')
print(f'Classes: {classes}')
print(f'Number of training images: {num_training_images}')


# call the model
model = models_vit.__dict__[base_model](
    num_classes=num_classes,
    drop_path_rate=drop_path,
    global_pool=True,
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
    'batch_size': 8,
    'epochs': 5,
    'model': 'vit_large_patch16',
    'base_learning_rate': '5e-5',
    'layer_decay': 0.65,
    'weight_decay': 0.05,
    'drop_path_rate': 0.2,
    'input_size': 224,
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
    '--nb_classes',str(num_classes),
    '--task', f'./{task}/',
    '--output_dir', output_folder,

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
    "epochs": epochs
}

# Content for the model card
model_card = {
    "model_name": f'{task}-{time}',
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

