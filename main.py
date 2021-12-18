from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Dense, InputLayer
from keras.models import Model
from keras.regularizers import l2
from sklearn.utils import validation
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from keras import Sequential
import numpy as np
from PIL import Image

from data_loader import MyDataLoader
from helper_funcs import *
from neural_nets import NeuralNets
from skimage import color

from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
import sys

import numpy as np
import matplotlib.pyplot as plt
import os
  
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K


def main():
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)

    norm = (128,128)
    dl = MyDataLoader("combine2", norm)
    nn = NeuralNets(norm)

    #dl.normalize_train_data(norm_size=norm)

    #X, y = dl.get_lab_data()

    #dl.numpy_dump(X, "X")
    #dl.numpy_dump(y, "y")

    X = dl.numpy_load("X")
    y = dl.numpy_load("y")


    X_lab = X
    y_lab = map_to(y)

    X_train, X_test, y_train, y_test = train_test_split(X_lab, y_lab, test_size = 0.3, random_state = 3)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.33, random_state = 3)

    input_shape = (128, 128, 1)
    batch_size = 32
    kernel_size = 3
    latent_dim = 256
    layer_filters = [64, 128, 256]
    layer_filters = layer_filters[::-1]

    inputs = Input(shape = input_shape)
    x = inputs
    for filters in layer_filters:
        x = Conv2D(filters = filters,
            kernel_size = kernel_size,
            strides = 2,
            activation ='relu',
            padding ='same')(x)
  
    shape = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim, name ='latent_vector')(x)
    encoder = Model(inputs, latent, name ='encoder')

    latent_inputs = Input(shape =(latent_dim, ), name ='decoder_input')
    x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    # stack of Conv2DTranspose(256)-Conv2DTranspose(128)-
    # Conv2DTranspose(64)
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters = filters,
                        kernel_size = kernel_size,
                        strides = 2,
                        activation ='relu',
                        padding ='same')(x)
    outputs = Conv2DTranspose(filters = 2,
                            kernel_size = kernel_size,
                            activation ='sigmoid',
                            padding ='same',
                            name ='decoder_output')(x)
    decoder = Model(latent_inputs, outputs, name ='decoder')

    autoencoder = Model(inputs, decoder(encoder(inputs)),
                    name ='autoencoder')

    autoencoder.compile(optimizer=RMSprop(), loss=MeanAbsoluteError())

    # model = nn.model_s()
    # model.compile(optimizer=RMSprop(), loss=MeanAbsoluteError())
    # (optimizer=Adam(learning_rate=decay_schedule), loss='mse')
    # model = load_model("model_d")
    

    num_epochs = 10

    # percent = 0.6
    # size = X_train.shape[0]
    # patience = 5
    # callback = EarlyStopping(monitor="val_loss", patience=patience)

    log_dir = "logs/fit/" + "autoencoder_1"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

 
    nr = 69
    test_image = X_test[nr]
    test_image = np.expand_dims(test_image, axis=0)
    minmax = np.zeros(shape=(1,1,2,2))
    minmax[0,0,0,0] = np.max(y[:,:,:,0])
    minmax[0,0,1,0] = np.min(y[:,:,:,0])
    minmax[0,0,0,1] = np.max(y[:,:,:,1])
    minmax[0,0,1,1] = np.min(y[:,:,:,1])
    percent = 0.6
    size = X_train.shape[0]
    for i in range(num_epochs):
        rand_array = np.random.choice(size, int(percent*size))
        X_bla = X_train[rand_array]
        y_bla = y_train[rand_array]
        autoencoder.fit(X_bla, y_bla, batch_size = 32, epochs = 5, validation_data=(X_val, y_val), verbose=1, callbacks=[tensorboard_callback])#, callbacks=[callback])
        save_model(autoencoder, "autoencoder_1", brave=True)

        pred = autoencoder.predict(test_image)
        save_images(test_image, map_from(pred, minmax), name="epoch", enhance=True, enumerate=i)
    
    rand_array = np.random.choice(X_test.shape[0], 10)
    pred = autoencoder.predict(X_test[rand_array])
    show_images(X_test[rand_array], map_from(pred, minmax))

if __name__ == '__main__':
    main()