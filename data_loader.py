import os
from os import walk
import shutil
import json
import numpy as np
from PIL import Image, ImageChops
from tqdm import tqdm
from skimage import color
import tensorflow as tf
import tensorflow_io as tfio

class MyDataLoader(object):   

    def __init__(self, dirName, norm_size, jsonName="cat_to_name.json", trainName="train"):
        self.dirName = dirName
        self.jsonName = jsonName
        self.trainName = trainName
        self.norm_size = norm_size

        self.rootPath  = os.getcwd()
        self.dataPath  = os.path.join(self.rootPath, self.dirName)

        self.trainPath = os.path.join(self.rootPath, self.trainName)
        self.XPath = os.path.join(self.trainPath, "X")
        self.failPath  = os.path.join(self.trainPath, "fail")
        self.yPath = os.path.join(self.trainPath, "y")
        

        with open(self.jsonName) as json_file:
            self.nameDict = json.load(json_file)
        
        

    def normalize_train_data(self, aspect_ratio = 1.5, tolerance = 0.5):

        try:
            shutil.rmtree(self.trainPath)
        except:
            pass

        os.mkdir(self.trainPath)
        os.mkdir(self.yPath)
        os.mkdir(self.failPath)

        for (_, dirNames, _) in walk(self.dataPath):
            if dirNames==[]: continue
            for category in tqdm(dirNames):
                readPath = os.path.join(self.dataPath, category)
                writePath = os.path.join(self.yPath, category)
                os.mkdir(writePath)

                for (_,_,fileNames) in walk(readPath):

                    for imgNumber, imgName in enumerate(fileNames):
                        img = Image.open(readPath+"\\"+imgName)
                        if self.__resizable(img.size, aspect_ratio, tolerance):
                            img = img.resize(self.norm_size)

                            # Really slow but some images have weird formats that make it  crash otherwise
                            try:
                                if not self.__is_greyscale(img):
                                    img.save(os.path.join(writePath, self.nameDict[category]+str(imgNumber)+".jpg"), "JPEG")
                            except:
                                try:
                                    img = img.convert("RGB")
                                    if not self.__is_greyscale(img):
                                        img.save(os.path.join(writePath, self.nameDict[category]+str(imgNumber)+".jpg"), "JPEG")
                                except:
                                    raise ValueError(f"Cannot save image {imgName}")
                                pass
                        else:
                            try:
                                img.save(os.path.join(self.failPath, self.nameDict[category]+str(imgNumber)+".jpg"), "JPEG")
                            except:
                                try:
                                    img = img.convert("RGB")
                                    img.save(os.path.join(self.failPath, self.nameDict[category]+str(imgNumber)+".jpg"), "JPEG")
                                except:
                                    print(f"Cannot save image {imgName}")
                                    pass
                                pass

    def all_to_one(self):
        onePath = os.path.join(self.yPath, "0")

        if not os.path.isdir(onePath):
            os.mkdir(onePath)

        for (_, dirNames, _) in walk(self.yPath):

            if dirNames==[]: continue
            for dirName in tqdm(dirNames):
                if dirName=="0": continue
                dirPath = os.path.join(self.yPath, dirName)
                for (_,_,fileNames) in walk(dirPath):
                    
                    for fileName in fileNames:
                        readPath = os.path.join(dirPath, fileName)
                        writePath = os.path.join(onePath, fileName)
                        shutil.move(readPath, writePath)
                        
                os.rmdir(dirPath)
                     

    def get_lab_data(self):
        lab = self.__read_lab(self.__dir_size(self.yPath))
        X = (np.expand_dims(lab[:,:,:,0], -1)/100).astype(np.float16)
        y = (lab[:,:,:,1:]/128).astype(np.float16)
        return X, y

    def get_classification_data(self):                 
        return self.__read_classification(self.__dir_size(self.XPath))   

    def get_rgb_data(self):                  
        return self.__read_rgb(self.__dir_size(self.XPath))


    def get_all_filenames(self):
        size = self.__dir_size(self.yPath)
        filenames = np.empty(shape=size, dtype=object)
        relativePath = os.path.join(self.trainName, "y")
        count = 0
        for (_, dirNames, _) in walk(self.yPath):

            for dirName in dirNames:
                readPath = os.path.join(self.yPath, dirName)

                for (_, _, names) in walk(readPath):

                    for name in names:
                        tempPath = os.path.join(dirName, name)
                        filenames[count] = os.path.join(relativePath, tempPath)
                        count = count + 1
        return filenames

    def convert_all_to_npz(self):

        if not os.path.isdir(self.yPath):
            print("No such directory")

        for (_, dirNames, _) in walk(self.yPath):

            for dirName in dirNames:
                dirPath = os.path.join(self.yPath, dirName)
                for (_,_,fileNames) in walk(dirPath):
                    
                    if fileNames == []: continue
                    for fileName in tqdm(fileNames):
                        readPath = os.path.join(dirPath, fileName)
                        image = tf.io.read_file(readPath)
                        image = tf.image.decode_jpeg(image)
                        image = tf.image.convert_image_dtype(image, tf.float16)
                        image = tfio.experimental.color.rgb_to_lab(image)
                        image = image.numpy()
                        image[:,:,0] = image[:,:,0]/100
                        image[:,:,1:] = (image[:,:,1:]/128+1)/2
                        os.remove(readPath)
                        fileName, _ = fileName.split(".jpg")
                        writePath = os.path.join(dirPath, fileName)
                        np.savez(writePath, image)

        

    
    def __resizable(self, img_size, aspect_ratio, tolerance):
        width = img_size[0]
        height = img_size[1]

        if width >= height:
            return aspect_ratio - tolerance <= width/height <= aspect_ratio + tolerance
        return aspect_ratio - tolerance <= height/width <= aspect_ratio + tolerance

    def __is_greyscale(self, img):
        if img.mode not in ("L", "RGB"):
            raise ValueError("Unsuported image mode")

        if img.mode == "RGB":
            rgb = img.split()
            if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0: 
                return False
            if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0: 
                return False
        return True

    def __dir_size(self, path):
        if(os.path.isdir(path)):
            size = 0
            for (_, dirNames, _) in walk(path):
                for dirName in dirNames:
                    readPath = os.path.join(path, dirName)
                    for (_, _, fileNames) in walk(readPath):
                        size = size + len(fileNames)
            return size
        else:
            print("Not a valid path to directory")
            return 0

    def __read_lab(self, size):
        lab = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1], 3), dtype=np.int8)
        iter = 0
        for (_, dirNames, _) in walk(self.yPath):
            if dirNames==[]: continue
            for dirName in tqdm(dirNames):
                readPath = os.path.join(self.yPath, dirName)
                for (_, _, fileNames) in walk(readPath):
                    for name in fileNames:
                        img = color.rgb2lab(Image.open(readPath+"\\"+name))
                        lab[iter] = np.array(img, dtype=np.int8)
                        iter = iter + 1
        return lab


    def __read_classification(self, size):
        X = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1]), dtype=np.uint8)
        y = np.ndarray(shape=size)
        iter = 0
        for (_, dirNames, _) in walk(self.XPath):
            if dirNames==[]: continue
            for dirName in tqdm(dirNames):
                readPath = os.path.join(self.XPath, dirName)
                for (_, _, fileNames) in walk(readPath):
                    for name in fileNames:
                        img = Image.open(readPath+"\\"+name)
                        X[iter] = np.array(img, dtype=np.uint8)
                        y[iter] = dirName
                        iter = iter + 1
        return X, y
    
    def __read_rgb(self, size):
        X = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1]), dtype=np.uint8)
        y = np.ndarray(shape=(size, self.norm_size[0], self.norm_size[1], 3), dtype=np.uint8)
        iter = 0
        for (_, dirNames, _) in walk(self.XPath):
            if dirNames==[]: continue
            for dirName in tqdm(dirNames):
                if dirName=="fail": continue
                XreadPath = os.path.join(self.XPath, dirName)
                yreadPath = os.path.join(self.yPath, dirName)

                tempIter = iter
                for (_, _, fileNames) in walk(XreadPath):
                    for name in fileNames:
                        img = Image.open(XreadPath+"\\"+name)
                        X[tempIter] = np.array(img, dtype=np.uint8)
                        tempIter = tempIter + 1

                tempIter = iter
                for (_, _, fileNames) in walk(yreadPath):
                    for name in fileNames:
                        img = Image.open(yreadPath+"\\"+name)
                        y[tempIter] = np.array(img, dtype=np.uint8)
                        tempIter = tempIter + 1
                iter = tempIter
        return X, y

    

# def train_test_val_split(self, ratio_test):
    #     try:
    #         shutil.rmtree(self.testPath)
    #     except:
    #         pass

    #     os.mkdir(self.testPath)

    #     for (_, dirNames, _) in walk(self.trainPath):
    #         for dirName in dirNames:
    #             newTrainDir = os.path.join(self.trainPath, dirName)
    #             newTestDir = os.path.join(self.testPath, dirName)
    #             os.mkdir(newTestDir)

    #             for (_, _, fileNames) in walk(newTrainDir):
    #                 for name in fileNames:
    #                     if random.random() < ratio_test:
    #                         oldPath = os.path.join(newTrainDir, name)
    #                         newPath = os.path.join(newTestDir, name)
    #                         shutil.move(oldPath, newPath)