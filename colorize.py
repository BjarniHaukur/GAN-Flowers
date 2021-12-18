import os
import numpy as np
from PIL import Image
from skimage import color
import tensorflow as tf
from helper_funcs import load_model, map_to, map_from



class Colorize(object):

    def __init__(self, image_path, output_size=None):
        
        self.image_path = image_path
        self.output_size = output_size
        self.old_image = None
        
    def __SSIMLoss(self, y_true, y_pred):
        return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)))#+0.5*tf.keras.metrics.mean_squared_error(y_true,y_pred)
        
    def __open_image(self):
        if os.path.isfile(self.image_path) \
        or os.path.isfile(os.path.join(self.image_path, ".png")) \
        or os.path.isfile(os.path.join(self.image_path, ".jpg")) \
        or os.path.isfile(os.path.join(self.image_path, ".jpeg")):

            image = Image.open(self.image_path)
            self.old_image = image
            self.output_size = image.size
            return image
        else:
            raise ValueError("Cannot read path to image")


    def predict(self):
        img = self.__open_image()
        img = img.resize((128,128))
        gray = color.rgb2lab(img)[:,:,0]
        gray = np.expand_dims(map_to(gray/100), 0)

        model = load_model("model_e40")#load_model("model_i_SSIMV_40", custom={'SSIMLoss':self.SSIMLoss()})

        pred = model(gray)

        canvas = np.zeros((128,128,3))
        canvas[:,:,0] = gray*100
        canvas[:,:,1:] = map_from(pred)*128
        canvas = (color.lab2rgb(canvas)*255).astype(np.uint8)

        out = Image.fromarray(canvas)
        out = out.resize(self.output_size, Image.ANTIALIAS)
        out.save('recolor.jpg')

