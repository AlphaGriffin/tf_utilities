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
                 f_size=5,
                 fc_size=128,
                 save_path="/home/eric/repos/pycharm_repos/tf_utilities/models/",
                 logdir="/home/eric/repos/pycharm_repos/tf_utilities/models/",
                 save_dataset=True,
                 ):

        self.verbose = verbose
        
        # learning rate has been DEPRICATED
        # brought it back
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # CNN options
        self.conv_layers = conv_layers  # of conv layers 2 stock
        self.fc_layers = fc_layers      # of fully connected layers 2 stock
        self.f_size = f_size            # fixed size not file size... silly 5 stock
        self.fc_size = fc_size          # Max # of elements in FC flatened layers
        self.save_path = save_path
        self.logDir = logdir

        # saves the input dataset to the logdir as a tf_record
        self.save_dataset = save_dataset
        
        # Mupen64plus stuff
        #self.save_path = "/home/eric/.local/share/mupen64plus/model/mariokart64/"
        # server port for webAPI model short range
        #self.port = 8321
