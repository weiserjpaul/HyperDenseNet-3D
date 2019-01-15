from keras.models import load_model
from Configurations import config
from src.load_data import load_data, get_files
from src.PredictSave import predict, save
from src.loss_functions import weighted_dice_coefficient_loss


paths, masks_paths = get_files(path_to_data = config["path_to_data"], 
                               train_modalities = config["train_modalities"])

images, masks, affines = load_data(paths = paths, masks_paths = masks_paths, 
                                   train_modalities = config["train_modalities"], 
                                   image_shape = config["image_shape"], 
                                   train_validate_rate = config["train_validate_rate"])

model = load_model(config["path_to_model"], 
                   custom_objects={'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss})

predictions_0, predictions_1, constr_masks = predict(model = model, images = images, masks = masks, 
                                                     input_shape = config["input_shape"], 
                                                     image_shape = config["image_shape"])

save(predictions = predictions_1, images = images, 
     constr_masks = constr_masks, affines = affines)
