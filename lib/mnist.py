#!/usr/bin/python
#
# Copyright (C) 2016 Alpha Griffin

"""  THIS HELPER WILL DEAL WITH THE MNIST DATASET   """

import tensorflow.contrib.learn.python.learn.datasets.mnist as fthedang
import numpy as np # like a boss

ROOT = '/home/eric/eric/practice/models/mnist/'

class MNIST_DATASET(object):
    def __init__(self, options):
        self.name                  = 'MNIST'                            # Mixed National Institute of Standards and Technology database
        self.options               = options                            # MASTER: options should have everything
        self.path                  = self.options.model_path + 'mnist/' # hard coded link from master... $ hopefully this lives
        self.build_return();                                            # This is the final building command for output
        self.img_size = img_size   = 28                                 # THis is KNOWN for this dataset
        self.num_classes           = 10                                 # THis is KNOWN for this dataset    
        self.num_channels          = 1                                  # ???? R G B ?? GRAY = 1
        self.img_size_flat         = img_size * img_size
        self.img_shape             = (img_size, img_size)
        self.batch_size            = self.options.batch_size
        assert self.train_images.shape[0] == self.train_labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (self.train_images.shape, self.train_labels.shape))
        self.number_of_examples    = self.train_labels.shape[0]
       
    def load_dataset(self,):
        return fthedang.read_data_sets(self.path, one_hot=True)
       
    def build_return(self):
        self.data = self.load_dataset()                        # call TF lib for loading their data
        #this is a bad holder... because its a setup call...        
        self.trainer = self.data.train ## THIS WORKS GREAT!
        self.tester = self.data.test
        self.validator = self.data.validation
        self.train_labels = self.data.train.labels             # Build the necessary output data... should be another func really...
        self.test_labels = self.data.test.labels               # need it
        self.validation_labels = self.data.validation.labels   # need it
        self.data.test.cls = np.array([label.argmax() for label in self.data.test.labels])
        self.data.validation.cls = np.array([label.argmax() for label in self.data.validation.labels])
        self.cls_true = self.test_cls = self.data.test.cls
        self.images_test = self.test_images = self.data.test.images    
        self.images_train = self.train_images = self.data.train.images           
        if self.options.verbose: print("Loaded MNIST dataset... should have real name here...")      
        