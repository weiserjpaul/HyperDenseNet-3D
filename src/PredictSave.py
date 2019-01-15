import numpy as np
import scipy.misc
import nibabel as nib
from src.Decompose import destruct, reconstruct


def predict(model, images, masks, input_shape = (1,32,32,32), image_shape = (210,238,200)):
    output_shape = tuple([int(i) for i in model.output.shape[1:]])
    predictions_0 = np.zeros(tuple([len(images)])+image_shape)
    predictions_1 = np.zeros(tuple([len(images)])+image_shape)
    constr_masks = np.zeros(tuple([len(images)])+image_shape)

    for i in range(len(images)):
        patches_image, patches_mask = destruct(images[i], masks[i],
                                               inputsz=input_shape, outputsz=output_shape)
        pred = model.predict(patches_image)
        predictions_0[i] = reconstruct(pred[:,0])
        predictions_1[i] = reconstruct(pred[:,1])
        constr_masks[i] = reconstruct(patches_mask)
    return predictions_0, predictions_1, constr_masks


def save(predictions, images, constr_masks, affines):
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    
    for i in range(len(predictions)):
        img = nib.Nifti1Image(predictions[i], affines[i])
        filename = "predictions/pred"+str(i)+".nii.gz"
        nib.save(img, filename)

        img = nib.Nifti1Image(images[i,0], affines[i])
        filename = "predictions/image"+str(i)+".nii.gz"
        nib.save(img, filename)

        img = nib.Nifti1Image(constr_masks[i], affines[i])
        filename = "predictions/mask"+str(i)+".nii.gz"
        nib.save(img, filename)

