#!/usr/bin/python
# Copyright (C) 2017 Alpha Griffin


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

    Returns
    -------
    This is returns no elements but is used to build

    Example
    -------
    >>> config = options(verbose=False, learning_rate=.05, fc_layers=2)
"""
    def __init__(self,
                 verbose=True,
                 learning_rate=1e-7,
                 batch_size=100,
                 save_path=None,
                 conv_layers=5,
                 fc_layers=5,
                 fc_size=128,
                 logdir=None):
        
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # CNN options
        self.conv_layers = 5      # of conv layers 2 stock
        self.fc_layers = 5        # of fully connected layers 2 stock
        self.f_size = 5           # fixed size not file size... silly 5 stock
        self.fc_size = 128        # Max # of elements in FC flatened layers
        self.L2NormConst = learning_rate
        
        """Use Hard Coded Personal Pathing"""
        # Save Model Path
        if save_path is None:
            save_path = "/home/eric/.local/share/mupen64plus/model/mariokart64/"
        self.save_path = save_path
        
        # LogDir for Supervisor Object
        if logdir is None:
            self.logdir = "/tmp/train_logs"
        self.logdir = logdir
