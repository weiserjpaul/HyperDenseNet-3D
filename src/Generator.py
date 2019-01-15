from keras.utils import to_categorical
import numpy as np
import random


def generator(images, masks, batchsz=10, input_shape=(1,32,32,32), output_shape=(2,14,14,14),
              img_shape=(210,238,200), train_modalities = ["image"]):
    while True:
        data = np.zeros((batchsz,input_shape[0],input_shape[1],input_shape[2],input_shape[3]))
        labels = np.zeros((batchsz,output_shape[1],output_shape[2],output_shape[3]))

        for i in range(batchsz):
            index_patient = random.randrange(len(masks))
            
            dim1, dim2, dim3 = return_coords_brain_and_non_brain(masks = masks,
                                                                 index_patient=index_patient, 
                                                                 input_shape=input_shape, 
                                                                 output_shape=output_shape, 
                                                                 img_shape=img_shape)
            
            data[i] = images[index_patient,:,
                             dim1:dim1+input_shape[1],
                             dim2:dim2+input_shape[2],
                             dim3:dim3+input_shape[3]]
            shift1 = int((input_shape[1]-output_shape[1])/2)
            shift2 = int((input_shape[2]-output_shape[2])/2)
            shift3 = int((input_shape[3]-output_shape[3])/2)
            labels[i] = masks[index_patient,
                              dim1+shift1:dim1+shift1+output_shape[1], 
                              dim2+shift2:dim2+shift2+output_shape[2],
                              dim3+shift3:dim3+shift3+output_shape[3]]
        labels = to_categorical(labels, num_classes=output_shape[0])
        labels = np.moveaxis(labels,-1,1)
        yield tuple([data, labels])
            

def return_coords_brain_and_non_brain(masks, index_patient, input_shape=(1,32,32,32),
                                      output_shape=(2,14,14,14), img_shape=(210,238,200)):
    dim1 = random.randrange(img_shape[0] - input_shape[1])
    dim2 = random.randrange(img_shape[1] - input_shape[2])
    dim3 = random.randrange(img_shape[2] - input_shape[3])
    shift1 = int((input_shape[1]-output_shape[1])/2)
    shift2 = int((input_shape[2]-output_shape[2])/2)
    shift3 = int((input_shape[3]-output_shape[3])/2)
    tmp_mask = masks[index_patient,
                     dim1+shift1:dim1+shift1+output_shape[1], 
                     dim2+shift2:dim2+shift2+output_shape[2],
                     dim3+shift3:dim3+shift3+output_shape[3]]
    
    while (not np.any(tmp_mask)) or np.all(tmp_mask):
        dim1 = random.randrange(img_shape[0] - input_shape[1])
        dim2 = random.randrange(img_shape[1] - input_shape[2])
        dim3 = random.randrange(img_shape[2] - input_shape[3])
        tmp_mask = masks[index_patient, 
                         dim1+shift1:dim1+shift1+output_shape[1], 
                         dim2+shift2:dim2+shift2+output_shape[2],
                         dim3+shift3:dim3+shift3+output_shape[3]]
    
    return dim1, dim2, dim3


def return_coords_all(masks, index_patient, input_shape=(1,32,32,32),
                      output_shape=(2,14,14,14), img_shape=(210,238,200)):
    dim1 = random.randrange(img_shape[0] - input_shape[1])
    dim2 = random.randrange(img_shape[1] - input_shape[2])
    dim3 = random.randrange(img_shape[2] - input_shape[3])
