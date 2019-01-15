from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.Generator import generator
from math import floor
import os

from src.Generator import generator



def train(model, images, masks, checkpoint_file, input_shape = (1,32,32,32), image_shape = (210,238,200), 
          train_validate_rate = 0.85, patience = 50, steps_per_epoch = 300, validation_steps = 50, epochs=500):
    output_shape = (int(model.output.shape[1]), int(model.output.shape[2]),
                    int(model.output.shape[3]), int(model.output.shape[3]))
    
    train_val_index = floor(train_validate_rate * len(images))

    training_generator = generator(images[:train_val_index], masks[:train_val_index],
                                   batchsz=20, input_shape=input_shape,
                                  img_shape=image_shape, output_shape=output_shape)
    
    validation_generator = generator(images[train_val_index:], masks[train_val_index:], 
                                     batchsz=10, input_shape=input_shape,
                                  img_shape=image_shape, output_shape=output_shape)

    if not os.path.exists(checkpoint_file):
        os.mkdir(checkpoint_file)
    cp = ModelCheckpoint(filepath = checkpoint_file + "/model.{epoch:02d}-{val_loss:.2f}.h5", save_best_only=True)
    es = EarlyStopping(patience = patience)
    model.fit_generator(generator = training_generator,
                        validation_data = validation_generator,
                        steps_per_epoch = steps_per_epoch,
                        validation_steps = validation_steps,
                        epochs = epochs,
                        callbacks = [es, cp])
    return model
