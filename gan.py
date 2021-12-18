import numpy as np
from tensorflow.python.keras.losses import MeanSquaredError
from tqdm import tqdm
import tensorflow as tf
from keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy
from IPython.display import clear_output

class MyGAN(object):

    # Based on https://www.tensorflow.org/tutorials/generative/dcgan

    def __init__(self, ds_train, ds_val, gen_model, disc_model):
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.gen = gen_model
        self.disc = disc_model

        self.gen_train_loss_history = []
        self.disc_train_loss_history = []

        self.gen_val_loss_history = []
        self.disc_val_loss_history = []
        
        self.gen_opt = tf.keras.optimizers.Adam()
        self.disc_opt = tf.keras.optimizers.Adam()

        self.pre_loss = MeanSquaredError()
        
        self.ce = tf.keras.losses.BinaryCrossentropy()

        self.gen_loss = 0.0
        self.disc_loss = 0.0
        self.gen_val_loss = 0.0
        self.disc_val_loss = 0.0

        self.gan_trained = False


    def is_trained(self):
        return self.gan_trained


    def __discriminator_loss(self, disc_real, disc_fake):
        real_loss = self.ce(tf.ones_like(disc_real), disc_real)
        fake_loss = self.ce(tf.zeros_like(disc_fake), disc_fake)
        return real_loss + fake_loss


    def __generator_loss(self, disc_fake):
        return self.ce(tf.ones_like(disc_fake), disc_fake)


    def train_step(self, train_batch):
        X_real = train_batch[0]
        y_real = train_batch[1]
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            y_fake = self.gen(X_real)

            disc_real = self.disc(y_real)
            disc_fake = self.disc(y_fake)

            gen_loss  = self.__generator_loss(disc_fake)
            disc_loss = self.__discriminator_loss(disc_real, disc_fake)

        gen_grad  = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        self.gen_opt.apply_gradients(zip(gen_grad, self.gen.trainable_variables))
        self.disc_opt.apply_gradients(zip(disc_grad, self.disc.trainable_variables))

        self.gen_loss = gen_loss
        self.disc_loss = disc_loss


    def validation(self, val_batch):
        X_real = val_batch[0]
        y_real = val_batch[1]

        y_fake = self.gen(X_real)

        disc_real = self.disc(y_real)
        disc_fake = self.disc(y_fake)

        self.gen_val_loss  = self.__generator_loss(disc_fake)
        self.disc_val_loss = self.__discriminator_loss(disc_real, disc_fake)

    def __log(self, epoch, train_stats=None, val_stats=None):
        print(f'___________________________ Epoch {epoch+1} ______________________')
        if not train_stats is None:
            print('Generator loss:         {:.3f}'.format(train_stats[0]))
        if not val_stats is None:
            print('Generator val loss:     {:.3f}'.format(val_stats[0]))
        if not train_stats is None:
            print('Discriminator loss:     {:.3f}'.format(train_stats[1]))
        if not val_stats is None:
            print('Discriminator val loss: {:.3f}'.format(val_stats[1]))
    
    
    def train_individually(self, gen_epochs, disc_epochs, num_train_steps, num_val_steps):
        if self.gan_trained:
            print("Cannot train generator and discriminator separately after GAN initialization")
            return
        
        self.gen.compile(optimizer='rmsprop', loss='mse')
        self.gen.fit(self.ds_train, epochs=gen_epochs, steps_per_epoch=num_train_steps, validation_data=self.ds_val, validation_steps=num_val_steps)


        for epoch in range(disc_epochs):
            for i, data in enumerate(self.ds_train):
                if i == num_train_steps: break

                with tf.GradientTape() as disc_tape:
                    y_fake = self.gen(data[0])

                    disc_real = self.disc(data[1])
                    disc_fake = self.disc(y_fake)

                    disc_loss = self.__discriminator_loss(disc_real, disc_fake)

                disc_grad = disc_tape.gradient(disc_loss, self.disc.trainable_variables)
                self.disc_opt.apply_gradients(zip(disc_grad, self.disc.trainable_variables))

            print(f'_______________________ Epoch {epoch+1} ________________________')
            print(f'Discriminator loss:     {disc_loss:.3f}, ')


    def fit(self, num_epochs, num_train_steps, num_val_steps):
        self.gan_trained = True

        for epoch in range(num_epochs):

            train_stats = np.zeros(shape=2)
            for i, data in enumerate(self.ds_train):
                if i == num_train_steps: break
                self.train_step(data)
                train_stats = train_stats + np.array([self.gen_loss, self.disc_loss])

            train_stats = train_stats/num_train_steps
 
            self.gen_train_loss_history.append(train_stats[0])
            self.disc_train_loss_history.append(train_stats[1])

            val_stats = np.zeros(shape=2)
            print("Validating...")
            for i, data in enumerate(self.ds_val):
                if i == num_val_steps: break
                self.validation(data)
                val_stats = val_stats + np.array([self.gen_val_loss, self.disc_val_loss])

            val_stats = val_stats/num_val_steps

            self.gen_train_loss_history.append(val_stats[0])
            self.disc_train_loss_history.append(val_stats[1])

            self.__log(epoch, train_stats, val_stats)

    def get_generator(self):
        if not self.gan_trained:
            print("Warning: Model not trained")
        return self.gen
            

    def get_current_history(self):
        """
        Returns gen_train_loss, disc_train_loss and
        gen_loss_loss, disc_loss_loss as two tuples
        """
        return (self.gen_train_loss_history, self.disc_train_loss_history), \
               (self.gen_val_loss_history, self.disc_val_loss_history)