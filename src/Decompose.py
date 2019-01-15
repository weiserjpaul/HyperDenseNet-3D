import numpy as np
import random
from math import ceil


def destruct(image, mask, imagesz = (210,238,200), inputsz = (1,32,32,32), outputsz = (2,14,14,14)):
    """The input is an image and a mask of shape (modalities, imagesz) and (imagesz).
    The method decompses it into patches of size (inputsz) and (outputsz[1-3]).
    Finally it return a numpy array of shape (number of patches, inputsz) and 
    (number of patches, 1, outputsz[1-3])"""

    cut_mask = np.zeros(imagesz)
    while not np.any(cut_mask):
        index = random.randrange(imagesz[0])
        cut_img = image[:,index]
        cut_mask = mask[index]

    n_height = ceil(imagesz[0]/outputsz[1])
    n_width = ceil(imagesz[1]/outputsz[2])
    n_depth = ceil(imagesz[2]/outputsz[3])
    n_patches = n_height * n_width * n_depth
    patches_mask = np.zeros((n_patches, outputsz[1],outputsz[2],outputsz[3]))
    patches_image = np.zeros((n_patches, inputsz[0], inputsz[1], inputsz[2],inputsz[3]))

    big_cut_mask = np.zeros((imagesz[0] + inputsz[1]*2, imagesz[1] + inputsz[2]*2,
                             imagesz[2] + inputsz[3]*2))
    big_cut_mask[inputsz[1]:inputsz[1]+imagesz[0], inputsz[2]:inputsz[2]+imagesz[1],
                 inputsz[3]:inputsz[3]+imagesz[2]] = mask
    big_cut_img = np.zeros((inputsz[0], imagesz[0] + inputsz[1]*2, imagesz[1] + inputsz[2]*2,
                            imagesz[2] + inputsz[3]*2))
    big_cut_img[:, inputsz[1]:inputsz[1]+imagesz[0], inputsz[2]:inputsz[2]+imagesz[1],
                inputsz[3]:inputsz[3]+imagesz[2]] = image

    for x in range(n_height):
        for y in range(n_width):
            for z in range(n_depth):
                j = x*outputsz[1]
                k = y*outputsz[2]
                l = z*outputsz[3]
                shift_j = int((inputsz[1]-outputsz[1])/2)
                shift_k = int((inputsz[2]-outputsz[2])/2)
                shift_l = int((inputsz[3]-outputsz[3])/2)
                patches_mask[x*n_width*n_depth+y*n_depth+z] = big_cut_mask[inputsz[1]+j:inputsz[1]+j+outputsz[1],
                                                                           inputsz[2]+k:inputsz[2]+k+outputsz[2],
                                                                           inputsz[3]+l:inputsz[3]+l+outputsz[3]]
                patches_image[x*n_width*n_depth+y*n_depth+z] = big_cut_img[:,inputsz[1]+j-shift_j:inputsz[1]+j+inputsz[1]-shift_j,
                                                                           inputsz[2]+k-shift_k:inputsz[2]+k+inputsz[2]-shift_k,
                                                                           inputsz[3]+l-shift_l:inputsz[3]+l+inputsz[3]-shift_l]

    patches_mask = np.reshape(patches_mask,(n_patches, 1, outputsz[1],outputsz[2], outputsz[3]))
    return patches_image, patches_mask


def reconstruct(patches, imagesz = (210, 238, 200), inputsz = (2,32,32,32), outputsz = (2,14,14,14)):
    """Input is an numpy array of size (number of patches, outputsz[1-3]), which
    represent a decomposition of an image into smaller patches. The method composes
    the patches back into the image. The shape of the output image is (imagesz[1-3])"""

    n_height = ceil(imagesz[0]/outputsz[1])
    n_width = ceil(imagesz[1]/outputsz[2])
    n_depth = ceil(imagesz[2]/outputsz[3])
    reconstr = np.zeros((n_height*outputsz[1], n_width*outputsz[2], n_depth*outputsz[3]))
    for x in range(n_height):
        for y in range(n_width):
            for z in range(n_depth):
                j = x*outputsz[1]
                k = y*outputsz[2]
                l = z*outputsz[3]
                reconstr[j:j+outputsz[1], k:k+outputsz[2],l:l+outputsz[3]] = patches[x*n_width*n_depth+y*n_depth+z]
    reconstr = reconstr[:imagesz[0],:imagesz[1],:imagesz[2]]
    return reconstr
