#!/usr/bin/python
# Copyright (C) 2017 Alpha Griffin

class options(object):
    """options are for building a CNN network with tensorflow"""
    def __init__(self, 
                 verbose=True,
                 learning_rate=1e-7, 
                 batch_size=100, 
                 optimizer=2, 
                 entropy=1,
                 ):
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.batch_size = batch_size 
        self.optimizer = optimizer
        self.entropy = entropy # depricated
        
        ## CNN options
        self.conv_layers = 5      # of conv layers 2 stock
        self.fc_layers = 5        # of fully connected layers 2 stock     
        self.f_size = 5           # fixed size not file size... silly 5 stock
        self.fc_size = 128        # Max # of elements in FC flatened layers
        self.L2NormConst = learning_rate
        
        self.save_path = "/home/eric/.local/share/mupen64plus/model/mariokart64/"