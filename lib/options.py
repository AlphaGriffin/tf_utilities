#!/usr/bin/python
# Copyright (C) 2017 Alpha Griffin

"""
DummyScript.com 2017
This is the options setup for running TF with some sanity.
"""
class options(object):
    """
    TF_utilities Options

    Parameters
    ----------
    verbose : boolean
      This really Turns up the output, nearly unreadable.
    learning_rate : 1e-7
      Bigger numbers have been known to show better results than this
    batch_size : 100
      Similar to picking lotto numbers, cant speak to how it effects the output
    conv_layers : 5
      Mnist set tutorial uses 2 conv
    fc_layers : 5
      Mnist set tutorial uses 2 FC
    f_size : 5
      fixed size of the conv_output
    fc_size : 128
      number of elements in the flattened layer ^ fc_layers
    save_path : "/home/eric/.local/share/mupen64plus/model/mariokart64/"
      output path of trained model for serving
    logdir : "/tmp/train_logs"
      output path of logs for tensorboard and tf.Supervisor

    Returns
    -------
    This is returns no elements but is used to build ML models

    Example
    -------
    >>> config = options(verbose=False, 
             learning_rate=.05, 
             fc_layers=2,
             save_path="/home/eric/.local/share/mupen64plus/model/mariokart64/",
             logdir="/tmp/train_logs")
"""
    def __init__(self,
                 verbose=True,
                 learning_rate=1e-7,
                 batch_size=100,
                 conv_layers=5,
                 fc_layers=5,
                 fc_size=128,
                 save_path=None,
                 logdir=None):
        
        self.verbose = verbose
        
        # learning rate has been DEPRICATED
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # CNN options
        self.conv_layers = 5      # of conv layers 2 stock
        self.fc_layers = 5        # of fully connected layers 2 stock
        self.f_size = 5           # fixed size not file size... silly 5 stock
        self.fc_size = 128        # Max # of elements in FC flatened layers
        
        """Use Hard Coded Personal Pathing"""
        # Save Model Path
        if save_path is None:
            save_path = "/home/eric/git/tf_utilities/models/"
        self.save_path = save_path
        
        # LogDir for Supervisor Object
        if logdir is None:
            self.logdir = "/home/eric/git/tf_utilities/train_logs/"
        self.logDir = logdir
        
        # Mupen64plus stuff
        #self.save_path = "/home/eric/.local/share/mupen64plus/model/mariokart64/"
        # server port for webAPI model short range
        #self.port = 8321
