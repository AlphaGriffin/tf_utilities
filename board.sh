#!/bin/bash
#"""
#DummyScript.com 2017
#Created on Tue Feb 28 00:01:38 2017
#@author: eric
#"""
#"""
#    Spyder Inspector Goods
#
#    Parameters
#    ----------
#    a : number
#      A number
#    b : number
#      Another number
#
#    Returns
#    -------
#    res : number
#      The average of a and b, computed using 0.5*(a + b)
#
#    Example
#    -------
#    >>> average(5, 10)
#    7.5
#"""
#"""
#usage: tensorboard [-h] [--logdir LOGDIR] [--debug [DEBUG]] [--nodebug]
#                   [--host HOST] [--inspect [INSPECT]] [--noinspect]
#                   [--tag TAG] [--event_file EVENT_FILE] [--port PORT]
#                   [--purge_orphaned_data [PURGE_ORPHANED_DATA]]
#                   [--nopurge_orphaned_data]
#                   [--reload_interval RELOAD_INTERVAL]
#"""

tensorboard --logdir ./train_logs --reload_interval 25