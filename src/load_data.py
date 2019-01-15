import nibabel as nib
import numpy as np
from math import floor
import os


def get_files(path_to_data, train_modalities):
    masks_paths = sorted(os.listdir(path_to_data + "Masks"))
    images_paths = sorted(os.listdir(path_to_data + "Images"))
    masks_paths = [path_to_data + "Masks/" + masks_paths[i] for i in range(len(masks_paths))]
    images_paths = [path_to_data + "Images/" + images_paths[i] for i in range(len(images_paths))]
    radial_dist_paths = len(images_paths)*[path_to_data + "coordinates/" + "radial_dist" + ".nii.gz"]
    # Add lists paths for different modalities here.
    # Make sure other lists have the name newModality_paths.
    paths = []
    for modality in train_modalities:
        tmp = modality + "_paths"
        paths.append(vars()[tmp])
    return paths, masks_paths



def load_data(paths, masks_paths, train_modalities, image_shape = (210,238,200), train_validate_rate = 0):
    """Loads Data from input paths and input masks_paths. Output are two numpy files
    of shape (number of paths, number of modalities, image_shape) and
    (number of paths, image_shape)"""
    
    train_val_index = floor(len(masks_paths)*train_validate_rate)
    masks_paths = masks_paths[train_val_index:]
    for i in range(len(paths)):
        paths[i] = paths[i][train_val_index:]

    images = np.zeros((len(paths[0]), len(train_modalities), image_shape[0], image_shape[1], image_shape[2]))
    affines = np.zeros((len(paths[0]), 4,4))
    print("loading images...")
    for i in range(len(paths[0])):
        print("..." + paths[0][i])
        for counter, c in enumerate(train_modalities):
            data = nib.load(paths[counter][i])
            affines[i] = data.affine
            data = data.get_fdata()
            images[i,counter] = data



    masks = np.zeros((len(masks_paths), image_shape[0], image_shape[1], image_shape[2]))
    print("loading masks...")
    for i in range(len(masks_paths)):
        print("..." + masks_paths[i])
        masks_data = nib.load(masks_paths[i])
        masks_data = masks_data.get_fdata()
        masks[i] = masks_data
    return images, masks, affines
