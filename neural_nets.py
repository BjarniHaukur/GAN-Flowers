
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2, l1
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from keras import Sequential

class NeuralNets(object):

        def __init__(self, norm_size):
                self.norm_size = norm_size
                self.input_shape = (norm_size[0], norm_size[1], 1)
                self.output_shape = (norm_size[0], norm_size[1], 2)

        def model_a(self, kernel = (3,3), sampling = (2,2)):
                num_nodes = 32

                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))
                model.add(Conv2D(num_nodes, kernel, activation='relu', padding='same'))
                model.add(Conv2D(num_nodes, kernel, activation='relu', padding='same', strides=2))
                model.add(Conv2D(num_nodes*2, kernel, activation='relu', padding='same'))
                model.add(UpSampling2D(sampling))
                model.add(Conv2D(num_nodes*2, kernel, activation='relu', padding='same', strides=2))
                model.add(Conv2D(num_nodes*4, kernel, activation='relu', padding='same'))
                model.add(Conv2D(num_nodes*4, kernel, activation='relu', padding='same', strides=2))
                model.add(Conv2D(num_nodes*2, kernel, activation='relu', padding='same'))
                model.add(UpSampling2D(sampling))
                model.add(Conv2D(num_nodes, kernel, activation='relu', padding='same'))
                model.add(UpSampling2D(sampling))
                model.add(Conv2D(2, kernel, activation='tanh', padding='same'))

                return model


        def model_b(self):
                return self.__ResNet34()


        def model_c(self, kernel=2):
                l2_reg = l2(1e-3)
                input_tensor = Input(shape=self.input_shape)
                x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_1', kernel_initializer="he_normal",
                        kernel_regularizer=l2_reg)(input_tensor)
                x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_2', kernel_initializer="he_normal",
                        kernel_regularizer=l2_reg, strides=(2, 2))(x)
                x = BatchNormalization()(x)

                x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_1', kernel_initializer="he_normal",
                        kernel_regularizer=l2_reg)(x)
                x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_2', kernel_initializer="he_normal",
                        kernel_regularizer=l2_reg,
                        strides=(2, 2))(x)
                x = BatchNormalization()(x)

                x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_1',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_2',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_3', kernel_initializer="he_normal",
                        strides=(2, 2))(x)
                x = BatchNormalization()(x)

                x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_1',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_2',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', name='conv4_3',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = BatchNormalization()(x)

                x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_1',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_2',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv5_3',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = BatchNormalization()(x)

                x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_1',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_2',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(512, (kernel, kernel), activation='relu', padding='same', dilation_rate=2, name='conv6_3',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = BatchNormalization()(x)

                x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_1',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_2',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv7_3',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = BatchNormalization()(x)

                x = UpSampling2D(size=(2, 2))(x)
                x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_1',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_2',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_3',
                        kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
                x = BatchNormalization()(x)
                x = (UpSampling2D((2,2)))(x)
                x = (Conv2D(64, kernel, activation='relu', padding='same'))(x)
                x = (UpSampling2D((2, 2)))(x)
                x = (Conv2D(32, kernel, activation='relu', padding='same'))(x)

                outputs = Conv2D(2, (1, 1), activation='sigmoid', padding='same', name='pred')(x)

                model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
                return model

        def model_d(self, kernel = (2,2)):
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='conv_1'))
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='conv_2', strides=2))
                model.add(Conv2D(128, kernel, activation='relu', padding='same', name='conv_3'))
                model.add(Conv2D(128, kernel, activation='relu', padding='same', name='conv_4', strides=2))
                model.add(Conv2D(256, kernel, activation='relu', padding='same', name='conv_5'))
                model.add(Conv2D(256, kernel, activation='relu', padding='same', name='conv_6', strides=2))
                model.add(Conv2D(512, kernel, activation='relu', padding='same', name='conv_7'))
                model.add(Conv2D(512, kernel, activation='relu', padding='same', name='conv_8', strides = 2))
                model.add(Conv2D(1024, kernel, activation='relu', padding='same', name='conv_9'))
                model.add(Conv2D(512, kernel, activation='relu', padding='same', name='conv_10'))
                model.add(Conv2D(256, kernel, activation='relu', padding='same', name='conv_11'))
                model.add(Conv2D(128, kernel, activation='relu', padding='same', name='conv_12'))
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='conv_13'))
                model.add(Conv2D(32, kernel, activation='relu', padding='same', name='conv_14'))
                model.add(Conv2D(2, kernel, activation='sigmoid', padding='same', name='conv_15'))
                return model

        def model_e(self):
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))
                model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
                # model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(UpSampling2D((2, 2)))
                # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(2, (3,3), activation='sigmoid', padding='same'))
                return model

        def model_f(self):
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(512, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(2, (3,3), activation='sigmoid', padding='same'))
                return model

        def model_g(self):
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(2, (3, 3), activation='sigmoid', padding='same'))
                model.add(UpSampling2D((2, 2)))
                return model
        
        def model_h(self):
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                model.add(Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
                model.add(Conv2DTranspose(2, (3, 3), activation='sigmoid', padding='same'))
                return model
                
        def model_i(self):
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                model.add(Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
                model.add(Conv2DTranspose(2, (3, 3), activation='sigmoid', padding='same'))
                return model

        def model_z(self):
                X = self.norm_size[0]
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))
                model.add(Conv2D(32, (3, 3), activation="relu", padding="same", strides=2))
                model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=2))
                model.add(MaxPool2D())
                model.add(MaxPool2D())
                # model.add(UpSampling2D((2, 2)))
                model.add(Flatten())
                model.add(Dense(64, activation="relu"))
                model.add(Dense(2048, activation="relu"))
                model.add(Dense(4096, activation="relu"))
                model.add(Dense((X ** 2), activation="relu"))
                model.add(Reshape((64, 64, -1)))
                model.add(Conv2D(2, (3, 3), activation="sigmoid", padding="same"))
                model.add(UpSampling2D((2, 2)))

                return model

        
        def autoencoder(self, filter_list=[64, 128, 256], hidden_size=256):
                """
                filter_list is an ordered, decreasing list of
                num_filters which is used for decoding and
                encoding (reversed)
                """
                # Encoder
                encoder_in = Input(shape=self.input_shape, name='encoder_in')
                x = encoder_in
                for i, filters in enumerate(filter_list):
                    x = Conv2D(filters = filters, kernel_size = (3, 3),
                               activation ='relu', padding ='same',
                               strides = 2, name=f'encoder{i+1}')(x)
                shape = K.int_shape(x)
                x = Flatten()(x)
                encoder_out = Dense(hidden_size, name='encoder_out')(x)
                encoder = Model(encoder_in, encoder_out, name='encoder')

                #Decoder
                decoder_in = Input(shape=(hidden_size, ), name='decoder_in')
                x = Dense(shape[1]*shape[2]*shape[3]) (decoder_in)
                x = Reshape((shape[1], shape[2], shape[3])) (x)
                for i, filters in enumerate(filter_list[::-1]):
                        x = Conv2DTranspose(filters = filters, kernel_size = (3, 3),
                                        activation ='relu', padding ='same',
                                        strides = 2, name=f'decoder{i+1}')(x)
                decoder_out = Conv2DTranspose(filters = 2, kernel_size = (3, 3),
                                        activation ='sigmoid', padding ='same',
                                        name='decoder_out')(x)
                decoder = Model(decoder_in, decoder_out, name='decoder')

                return Model(encoder_in, decoder(encoder(encoder_in)), name='autoencoder')

        def model_s(self):
                # X = self.norm_size[0]
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))

                model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
                # model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))

                model.add(Conv2D(128, (3, 3), activation="relu", padding="same", strides=2))
                # model.add(Conv2D(64, (3, 3), activation="relu", padding="same", strides=2))
                model.add(MaxPool2D())
                model.add(MaxPool2D())
                # model.add(UpSampling2D((2, 2)))
                model.add(Flatten())
                model.add(Dense(64, activation="relu"))
                model.add(Dense(4096, activation="relu"))
                model.add(Reshape((64, 64, -1)))
                model.add(UpSampling2D((2,2)))
                # model.add(Reshape((X, X, -1)))
                model.add(Conv2D(2, (3, 3), activation="sigmoid", padding="same"))

                return model


        def generator_a(self):
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))
                model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
                # model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(UpSampling2D((2, 2)))
                # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(2, (3,3), activation='sigmoid', padding='same'))
                return model
                
        
        def generator_b(self):
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
                model.add(Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
                model.add(Conv2DTranspose(2, (3, 3), activation='sigmoid', padding='same'))
                return model

        def discriminator_a(self):
                model = Sequential()
                model.add(InputLayer(input_shape=self.output_shape))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
                model.add(Dropout(0.5))
                model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
                model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(1, activation='sigmoid'))
                return model

        def discriminator_b(self):
                model = Sequential()
                model.add(InputLayer(input_shape=self.output_shape))
                model.add(Conv2D(64, (3, 3), activation='leaky_relu', padding='same', strides=2))
                model.add(Conv2D(128, (3, 3), activation='leaky_relu', padding='same', strides=2))
                model.add(Conv2D(256, (3, 3), activation='leaky_relu', padding='same', strides=2))
                model.add(Conv2D(512, (3, 3), activation='leaky_relu', padding='same'))
                model.add(Conv2D(1, (4, 4), activation='sigmoid'))
                return model

        def __res_block(self, inputs, filters=64):
                x = Conv2D(filters, (3, 3), padding='same')(inputs)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = Conv2D(filters, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = Add()([x, inputs])
                x = ReLU()(x)
                return x


        def __ResNet34(self):
                img_input = Input(shape=(self.norm_size[0], self.norm_size[1], 1))

                x = Conv2D(64, (7, 7), strides=2, padding='same')(img_input)
                x = MaxPooling2D()(x)

                x = self.__res_block(x, filters=64)
                x = self.__res_block(x, filters=64)
                x = self.__res_block(x, filters=64)

                x = Conv2D(128, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = MaxPooling2D()(x)
                x = Conv2D(128, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = self.__res_block(x, filters=128)
                x = self.__res_block(x, filters=128)
                x = self.__res_block(x, filters=128)

                x = Conv2D(256, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = MaxPooling2D()(x)
                x = Conv2D(256, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = self.__res_block(x, filters=256)
                x = self.__res_block(x, filters=256)
                x = self.__res_block(x, filters=256)
                x = self.__res_block(x, filters=256)
                x = self.__res_block(x, filters=256)

                x = Conv2D(512, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = MaxPooling2D()(x)
                x = Conv2D(512, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = ReLU()(x)
                x = self.__res_block(x, filters=512)
                x = self.__res_block(x, filters=512)

                return Model(img_input, x)

        def autoencoder_1(self, kernel=(3,3)):
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape, name='input'))
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='encode_2', strides=2))
                model.add(Conv2D(128, kernel, activation='relu', padding='same', name='encode_3', strides=2))
                model.add(Conv2D(256, kernel, activation='relu', padding='same', name='encode_4', strides=2))
                model.add(Flatten())

                model.add(Conv2D(256, kernel, activation='relu', padding='same', name='decode_1', strides=2))
                model.add(Conv2D(128, kernel, activation='relu', padding='same', name='decode_2', strides=2))
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='decode_3', strides=2))
                model.add(Conv2D(2, kernel, activation='sigmoid', padding='same', name='ouput'))
                return model

        def autoencoder_2(self, kernel=(3,3)):
                model = Sequential()
                model.add(InputLayer(input_shape=self.input_shape, name='input'))
                model.add(Conv2D(128, kernel, activation='relu', padding='same', name='encode_1', strides=(1,1)))
                model.add(BatchNormalization())
                model.add(Conv2D(128, kernel, activation='relu', padding='same', name='encode_2', strides=(1,1)))
                model.add(BatchNormalization())
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='encode_4', strides=(2,2)))
                model.add(BatchNormalization())
                model.add(UpSampling2D((2, 2)))
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='decode_1', strides=(2,2)))
                model.add(BatchNormalization())
                model.add(Conv2D(64, kernel, activation='relu', padding='same', name='decode_2', strides=(2,2)))
                model.add(Conv2D(32, kernel, activation='relu', padding='same', name='decode_3', strides=(2,2)))
                model.add(Conv2D(2, kernel, activation='sigmoid', padding='same', name='ouput'))
                model.add(BatchNormalization())
                return model