# Script to split the dataset into training and testing sets

import os
import glob
import shutil
import numpy as np

train_path = 'plants/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
new_train_path = 'new_' + train_path
new_test_path = new_train_path.replace('train', 'test')

subfolders = [ f.path for f in os.scandir(train_path) if f.is_dir() ]
test_percent = 0.2

for folder in sorted(subfolders):

    # For each subfolder, create a dictionary with the key as the image name
    # and the value as a list of all the filenames that contain that key
    filenames = sorted(glob.glob(os.path.join(folder, '*')))
    if 'Corn_(maize)___Common_rust_' in folder:
        keys = [f.split('/')[-1][:12] for f in filenames]
    else:
        keys = [f.split('/')[-1].split('_')[0] for f in filenames]
    keys = np.unique(keys)

    image_dict = {}

    # Create a dictionary with the key as the image name and the value as a list of all the filenames that contain that key
    for key in keys:
        image_dict[key] = []
        for filename in filenames:
            if key in filename:
                image_dict[key].append(filename)

    n_images = len(keys)
    print(f'There are {len(keys)} unique images in the folder {folder}')

    n_train = int(n_images * (1 - test_percent))
    n_test = n_images - n_train

    idx = np.arange(n_images)
    np.random.shuffle(idx)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    # Create the train and test folders
    train_folder = os.path.join(new_train_path, os.path.basename(folder))
    test_folder = os.path.join(new_test_path, os.path.basename(folder))
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    print(f'Creating {train_folder} and {test_folder}')
    # Copy the files to the train and test folders
    for i in train_idx:
        for filename in image_dict[keys[i]]:
            shutil.copy(filename, train_folder)
    for i in test_idx:
        for filename in image_dict[keys[i]]:
            shutil.copy(filename, test_folder)

# Copy the validation folder
# shutil.copytree('plants-disease-dataset/valid', 'new-plants-disease-dataset/valid')


old_train_filenames = sorted(glob.glob(os.path.join(train_path, '*/*')))
train_filenames = sorted(glob.glob(os.path.join(new_train_path, '*/*')))
test_filenames = sorted(glob.glob(os.path.join(new_test_path, '*/*')))

print('Before: train images:', len(old_train_filenames))
print('After: train images:', len(train_filenames))
print('After: test images:', len(test_filenames))

if len(old_train_filenames) != len(train_filenames) + len(test_filenames):
    print('Error: the number of images in the old train folder is not equal to the sum of the new train and test folders')
else:
    print('Success: the number of images in the old train folder is equal to the sum of the new train and test folders')