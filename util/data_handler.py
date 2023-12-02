import os
import shutil
from sklearn.model_selection import train_test_split
import traceback
import datetime

def split_dataset(parent_folder, train_size=0.7, val_size=0.15, test_size=0.15):

    #exclude dirs that start with dataset from classes
    classes = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f)) and not f.startswith('_dataset')]
    time = datetime.datetime.now().strftime("%m-%d-%Y-%H%M%S")

    output_folder = os.path.join(parent_folder, f'_dataset_{time}', 'training_data')
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

        try:

            for img in train:
                try:
                    if not os.path.exists(os.path.join(output_folder, 'train', cls)):
                        os.makedirs(os.path.join(output_folder, 'train', cls))
                    # copy the image to the train folder not move
                    shutil.copy(os.path.join(cls_path, img), os.path.join(output_folder, 'train', cls, img))

                except:
                    traceback.print_exc()
                    print(f'Error moving {img}')
            for img in val:
                try:
                    if not os.path.exists(os.path.join(output_folder, 'val', cls)):
                        os.makedirs(os.path.join(output_folder, 'val', cls))
                    # copy the image to the val folder not move
                    shutil.copy(os.path.join(cls_path, img), os.path.join(output_folder, 'val', cls, img))
                except:
                    print(f'Error moving {img}')
            for img in test:
                try:
                    if not os.path.exists(os.path.join(output_folder, 'test', cls)):
                        os.makedirs(os.path.join(output_folder, 'test', cls))
                    # copy the image to the test folder not move
                    shutil.copy(os.path.join(cls_path, img), os.path.join(output_folder, 'test', cls, img))
                except:
                    print(f'Error moving {img}')
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
