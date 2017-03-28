#!/usr/bin/env python3
"""
DummyScript.com 2017
Created on Sat Feb 25 21:34:07 2017
@author: eric
"""
import os
import numpy as np
from PIL import Image
"""
This prepares a[list] of mupen64plus game recorder directories. into a pair of
numpy bin objects called images.npy and labels.npy.
"""

class Prepare(object):
    """ 
    This will be used by both the Process and Playback Modules for conversion
    of images for Tensorflow.
    
    """
    def __init__(self,options=None):
        self.options = options
        self.selection = None
        self.root_dir = None
        self.currentGame = None
    
        
                
    def make_BW(self, rgb):
        """ 
        This is the "rec601 luma" algorithm to compute 8-bit greyscale
        Parameters
        ----------
            img : numpy img
        
        Returns
        -------
            img : numpy image
        
        Example
        -------
        >>> img = Prepare.make_BW(img)
        """
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    def prepare_image(self, img, makeBW=False):
        """ 
        This resizes the image to a tensorflowish size 
        Parameters
        ----------
            img : path to an image file
        
        Returns
        -------
            img : numpy image
        
        Example
        -------
        >>> img = Prepare.prepare_image(img, makeBW=True)
        """
        pil_image = Image.open(img)                       # open img
        x = pil_image.resize((200, 66), Image.ANTIALIAS)  # resizes image
        numpy_img = np.array(x)                           # convert to numpy 
        if makeBW:  
            numpy_img = self.make_BW(numpy_img)           # grayscale
        return numpy_img
        
    def processing(self, folders="", makeBW=False, save=True):
        """
        This has 3 steps, move files, convert images, create bins
        Parameters
        ----------
            folders : [list] 
                of paths to mupen64 game recorder directories.
                
        
        Returns
        -------
            dataset_x : a numpy dataset obect for X-input tensor object.
            dataset_y : a numpy dataset object for Y-label tensor object.
              Two objects that can be loaded by the mupen64 dataset TF_utilities
              library.
        
        Example
        -------
        >>> X = ["/path/","/path/","/path"]
        >>> X_images_load, Y_labels_load = Prepare.processing(X, makeBW=False, save=False)
        """
        """
        if folders is "": 
            folders = self.selection
        saveDir = os.path.join(self.root_dir, "datasets", self.currentGame)
        if not os.path.isdir(saveDir):
            os.mkdir(saveDir)
        datasetIndex = len(os.listdir(saveDir))
        dataset_x = []
        dataset_y = []
        datasetFilename_x = "_{}_dataset_{}_image.npy".format(
                                            self.currentGame,datasetIndex)
        datasetFilename_y = "_{}_dataset_{}_label.npy".format(
                                            self.currentGame,datasetIndex)
        """
        # for each folder given...
        for i in folders:
            current_path = os.path.join(self.work_dir,self.currentGame,i)
            labels, imgs = self.gamepadImageMatcher(current_path)
            dataset_y.append(labels)
            
            for image in imgs:
                img = self.prepare_image(os.path.join(current_path,image, makeBW))
                dataset_x.append(img)
        # complete the transfer
        dataset_x = np.asarray(dataset_x)
        dataset_y = np.concatenate(dataset_y)
        # save the files off
        if save:
            np.save(os.path.join(saveDir, datasetFilename_x), dataset_x)
            np.save(os.path.join(saveDir, datasetFilename_y), dataset_y)
        
        # return them in a pipline
        return dataset_x, dataset_y
    
    def gamepadImageMatcher(self, path):
        """
        - SAW - matches gamepad csv data rows to images based on timestamps
        Parameters
        ----------
            folder : "/path/"
                a single path with timestamped images and a timestamped labels
                csv file where the timestamp is in col[0] then any size(conservitive)
                
        Returns
        -------
            keep_imgs : a numpy dataset obect for saving a *.npy bin file.
            keep_labels : a numpy dataset object for saving a *.npy bin file.
              2 object that can be concatonated into very large binary save
              file objects for later ML use.
        
        Example
        -------
        >>> current_path = os.path.join(working_dir, i)
        >>> labels, imgs = self.gamepadImageMatcher(current_path)
        """
                
        # Open CSV for reading
        csv_path = os.path.join(path, "data.csv")
        csv_io = open(csv_path, 'r')
        
        # Convert to a true array
        csv = []
        for line in csv_io:
            # Split the string into array and trim off any whitespace/newlines
            csv.append([item.strip() for item in line.split(',')])
        if not csv:
            #print ("CSV HAS NO DATA")
            return None, None
            
        # Get list of images in directory and sort it
        all_files = os.listdir(path)
        images = []
        for filename in all_files:
            if filename.endswith('.png'):
                images.append(filename)
        images = sorted(images)
        
        if not images:
            #print ("FOUND NO IMAGES");
            return None, None
    
        # We're going to build up 2 arrays of matching size:
        keep_csv = []
        keep_images = []
    
        # Prime the pump (queue)...
        prev_line = csv.pop(0)
        prev_csvtime = int(prev_line[0])
    
        while images:
            imgfile = images[0]
            # Get image time:
            #     Cut off the "gamename-" from the front and the ".png"
            hyphen = imgfile.rfind('-') # Get last index of '-'
            if hyphen < 0:
                break
            imgtime = int(imgfile[hyphen+1:-4]) # cut it out!
            lastKeptWasImage = False # Did we last keep an image, or a line?
            if imgtime > prev_csvtime:
                keep_images.append(imgfile)
                del images[0]
                lastKeptWasImage = True
                
                # We just kept an image, so we need to keep a
                #corresponding input row too
                while csv:
                    line = csv.pop(0)
                    csvtime = int(line[0])
    
                    if csvtime >= imgtime:
                        # We overshot the input queue... ready to
                        # keep the previous data line
                        # truncate  the timestamp
                        keep_csv.append(prev_line[1:]) 
                        lastKeptWasImage = False
    
                        prev_line = line
                        prev_csvtime = csvtime
    
                        if csvtime >= imgtime:
                            break;
    
                    if not csv:
                        if lastKeptWasImage:
                            # truncate off the timestamp
                            keep_csv.append(prev_line[1:]) 
                        break
    
            else:
                del images[0]
        return keep_csv, keep_images