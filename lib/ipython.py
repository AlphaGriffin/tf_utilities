#!/usr/bin/python
#
# Copyright (C) 2016 Alpha Griffin && HVASS LABS

"""  THIS HELPER WILL DEAL WITH THE IPYTHON PLOTTING   """
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as tqdm_notebook
import math

def __init__():
    pass

class ipython(object):
    def __init__(self):
        self.test = 1
        pass
    
    def progress(self):
        return tqdm_notebook
        
class hvass_ipython(object):
    def __init__(self, model=None, network=None):
        #self.utils = utils
        self.model = model
        self.network = network
        if self.model is not None:
            self.options = model.options
        if self.network is not None:
            self.model = self.network.model
            self.options = self.network.options
        self.plt = plt
        self.test = 1
    
    # works for mnistdataset plot...
    def simple_plot(self, images, cls_true, cls_pred=None, logits=None, smooth=True):    
        assert len(images) == len(cls_true) == 9
        # Create figure with 3x3 sub-plots.
        fig, axes = self.plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.6, wspace=0.3)
        if self.model.img_size is None or 'NoneType': img_size = 28
        if smooth: # Interpolation type.
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
        img_shape = (img_size, img_size)
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i].reshape(img_shape), cmap='binary', interpolation=interpolation)
    
            # Show true and predicted classes.
            xlabel = "True: {0}".format(cls_true[i])
            if cls_pred is not None:
                #xlabel = "True: {0}".format(cls_true[i])
            #else:
                xlabel += ", Pred: {0}".format(cls_pred[i])
            if logits is not None:
                xlabel = ", Logits: {0}".format(cls_true[i], logits[i])
            ax.set_xlabel(xlabel)
            
            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])
        self.plt.show()
    
    
    def plot_images(self, images, cls_true, class_names, cls_pred=None, smooth=False):
        assert len(images) == len(cls_true)
        fig, axes = self.plt.subplots(3, 3) # Create figure with sub-plots.
        if cls_pred is None: # Adjust vertical spacing.
            hspace = 0.3
        else:
            hspace = 0.6
        fig.subplots_adjust(hspace=hspace, wspace=0.3)
        if smooth: # Interpolation type.
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'
    
        for i, ax in enumerate(axes.flat): # There may be less than 9 images, ensure it doesn't crash.
            if i < len(images):
                ax.imshow(images[i], interpolation=interpolation) # Plot image.
                cls_true_name = class_names[cls_true[i]] # Name of the true class.                
                if cls_pred is None: # Show true and predicted classes.
                    xlabel = "True: {0}".format(cls_true_name)
                else:
                    # Name of the predicted class.
                    cls_pred_name = class_names[cls_pred[i]]
    
                    xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)
                ax.set_xlabel(xlabel) # Show the classes as the label on the x-axis.
            ax.set_xticks([]) # Remove ticks from the plot.
            ax.set_yticks([])
        self.plt.show()
        
    def plot_transfer(self, i, transfer_values, image_set):
        print("Input image:")       
        self.plt.imshow(image_set[i], interpolation='nearest')
        self.plt.show()

        print("Transfer-values for the image using Inception model:")
        img = transfer_values[i]
        img = img.reshape((32, 64))
        self.plt.imshow(img, interpolation='nearest', cmap='Reds')
        self.plt.show()

    def plot_example_errors(self, images_test, cls_pred, correct):
        print("Loading Model Data...")
        m = self.model
        #images_test = m.images_test
        cls_test = m.cls_true
        incorrect = (correct == False)
        
        images = images_test[incorrect]
        cls_pred = cls_pred[incorrect]
        cls_true = cls_test[incorrect]
        
        n = min(9, len(images))
        self.simple_plot(images=images[0:n], cls_true=cls_true[0:n], cls_pred=cls_pred[0:n])
        
    def plot_weights(self, w):        
        w_min = np.min(w)
        w_max = np.max(w)
        fig, axes = plt.subplots(3, 4)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        if self.model.img_size is None or 'NoneType': img_size = 28;
        else: img_size = self.model.img_size;
        img_shape = (img_size, img_size)
        for i, ax in enumerate(axes.flat):
            if i<10:
                image = w[:, i].reshape(img_shape)
                ax.set_xlabel("Weights: {0}".format(i))
                ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
                 
    def plot_confused(self, cm, num_classes):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # Make various adjustments to the plot.
        plt.tight_layout()
        print(cm)        
        plt.matshow(cm)
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
       
    def plot_conv_weights(self, weights, input_channel=0): 
        w = weights
        w_min = np.min(w)
        w_max = np.max(w)
        print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
        num_filters = w.shape[3]
        num_grids = math.ceil(math.sqrt(num_filters))
        fig, axes = plt.subplots(num_grids, num_grids)
        for i, ax in enumerate(axes.flat):
            # Only plot the valid filter-weights.
            if i<num_filters:
                # Get the weights for the i'th filter of the input channel.
                # See new_conv_layer() for details on the format
                # of this 4-dim tensor.
                img = w[:, :, input_channel, i]
                ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
        
    def plot_image(self, image):
        if self.model.img_shape is None or 'NoneType': img_size = 28;
        else: img_size = self.model.img_shape
        img_shape = (img_size, img_size)        
        self.plt.imshow(image.reshape(img_shape),
                        interpolation='nearest',
                        cmap='binary')    
        self.plt.show()
    
    def plot_conv_layer(l):
        values = l
        num_filters = values.shape[3]
        num_grids = math.ceil(math.sqrt(num_filters))
        fig, axes = plt.subplots(num_grids, num_grids)
        for i, ax in enumerate(axes.flat):
            if i<num_filters:
                img = values[0, :, :, i]
                ax.imshow(img, interpolation='nearest', cmap='binary')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()        
    """             NEXT THING !!!         """      
         
        
        
        