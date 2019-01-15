from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation
from keras.layers import Softmax
from keras.layers import Concatenate, SpatialDropout3D, Conv3D, Cropping3D
from keras.engine import Model
from keras.optimizers import Adam

from src.loss_functions import weighted_dice_coefficient_loss


def HyperDenseNet(kernelshapes, numkernelsperlayer, input_shape, activation_name="sigmoid", dropout_rate=0.3, n_labels=2, 
                  optimizer=Adam, initial_learning_rate=5e-4, loss_function="categorical_crossentropy"):
    n_conv_layer = 0
    for kernel in kernelshapes:
        if len(kernel) == 3:
            n_conv_layer += 1
    layers = []

    inputs = Input(input_shape)
    current_layer = inputs
    layers.append(current_layer)

    for i in range(n_conv_layer):
        current_layer = Conv3D(numkernelsperlayer[i], kernelshapes[i], strides=(1,1,1), padding='valid', activation=activation_name, data_format='channels_first')(current_layer)
        layers.append(current_layer)
        cropped_layers = []
        n_layers = len(layers)
        for count, layer in enumerate(layers):
            cropped_layer = Cropping3D(cropping=(n_layers-1-count), data_format="channels_first")(layer)
            cropped_layers.append(cropped_layer)
        current_layer = Concatenate(axis = 1)(cropped_layers)

    for i in range(n_conv_layer, len(kernelshapes)):
        current_layer = Conv3D(numkernelsperlayer[i], [1,1,1], strides=(1,1,1), padding='valid', activation=activation_name, data_format='channels_first')(current_layer)
        current_layer = SpatialDropout3D(rate=dropout_rate, data_format='channels_first')(current_layer)

    current_layer = Conv3D(n_labels, [1,1,1], strides=(1,1,1), padding="valid", activation=None, data_format='channels_first')(current_layer)
    current_layer = Softmax(axis=1)(current_layer)

    model = Model(inputs = inputs, outputs = current_layer)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model
