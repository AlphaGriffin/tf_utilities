#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 18:28:20 2017

@author: eric
"""

import numpy as np # like a boss
from PIL import Image  # for more image manipulation

ROOT = "/home/eric/git/tf_utilities/"

class TKART_DATASET(object):
    """Build out a dataset of the TKART samples for training"""
    def __init__(self, options):
        self.name                  = 'TKART'                            # Mixed National Institute of Standards and Technology database
        self.options               = options                            # MASTER: options should have everything
        
        self.image_h = 66
        self.image_w = 200
        self.num_channels = 1     # yeah... just the h, w.. no one need i think   
        self.num_classes = 5  # one for each used button on the pad
        self.img_size_flat = self.image_h * self.image_w # FIXME CHANGE 3 to 1 if I get black and white images
        self.image_size = (self.image_w, self.image_h)
        self.batch_size = self.options.batchsize
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
        
    def build_return(self):
        img_list, pad = self.load_numpy_dataset()
        # test_set,train_set,vaildation_set = 
        
    def load_numpy_dataset(self,): 
        self._X = np.load("data/X.npy")
        self._y = np.load("data/y.npy")
        return self._X, self._y

    def prepare_numpy_dataset(self,dir=None): 
        """A proper pickle would probably be better for this task"""  # long term anyway
        # FIXME! check if Dir is None ... Pass this Shiz and see if this is done already
        if self.options.verbose: print("Processing data")
        X = []  # these are marked for their import in TF... this is the X_image
        y = []  # this is the = y_input_true or the Label for the given 
        
        # for each item in the list of dirs provided, create a single dataset
        for i in dir:
            if self.options.verbose: print("processing folder: {}".format(i))
            img_list, pad = self.load_sample(i)
    
            # add joystick values to y
            y.append(pad)
    
            # load, prepare and add images to X
            for j in img_list:
                #image = imread(image_file)  # this is using SKLEARN to open the img... is this the best solution?
                vec = self.prepare_image(j)  # this returns a BW numpy array 1-D object
                X.append(vec)                # add this to the list    
    
        if self.options.verbose: print("Attempting Save of Numpy Dataset")
        X = np.asarray(X)      # this creates a new np.array that is the list of images from all DIRS
        y = np.concatenate(y)  # read all vales
    
        np.save("data/X", X)
        np.save("data/y", y)
    
        if self.options.verbose: print("DataSet Prepared and Ready for Training")
        return True

        
    def make_BW(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    def prepare_image(self, img):
        pil_image = Image.open(img)                                         # open img
        pil_image.thumbnail((self.image_w, self.image_h), Image.ANTIALIAS)  # resize
        numpy_img = np.array(pil_image)                                     # convert to numpy
        grey_pil_image = self.make_BW(numpy_img)                            # reduce to BW
        return grey_pil_image

    @property
    def train_labels(self):
        return self._num_examples

    
    def batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._X[start:end], self._y[start:end]

    def load_sample(self, folder):
        image_files = np.loadtxt(folder + '/data.csv', delimiter=',', dtype=str, usecols=(0,))
        joystick_values = np.loadtxt(folder + '/data.csv', delimiter=',', usecols=(1,2,3,4,5))
        return image_files, joystick_values
        