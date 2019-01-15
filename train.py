from keras.optimizers import Adam
from src.load_data import load_data, get_files
from src.HyperDenseModel import HyperDenseNet
from src.Training import train

from Configurations import config


paths, masks_paths = get_files(path_to_data = config["path_to_data"], 
                               train_modalities = config["train_modalities"])

images, masks, affines = load_data(paths = paths, masks_paths = masks_paths, 
                                   train_modalities = config["train_modalities"], 
                                   image_shape = config["image_shape"], 
                                   train_validate_rate = 0)

model = HyperDenseNet(kernelshapes = config["kernelshapes"], 
                      numkernelsperlayer = config["numkernelsperlayer"],
                      input_shape = config["input_shape"],
                      activation_name = config["activation_name"],
                      dropout_rate = config["dropout_rate"],
                      n_labels = config["n_labels"],
                      optimizer = config["optimizer"],
                      initial_learning_rate = config["initial_learning_rate"],
                      loss_function = config["loss_function"])

model = train(model, images, masks,
              checkpoint_file = config["checkpoint_file"],
              input_shape = config["input_shape"], 
              image_shape = config["image_shape"],
              train_validate_rate = config["train_validate_rate"], 
              patience = config["patience"], 
              steps_per_epoch = config["steps_per_epoch"], 
              validation_steps = config["validation_steps"], 
              epochs = config["epochs"])


model.save(config["path_to_model"])
