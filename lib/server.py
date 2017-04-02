#!/usr/bin/env python
# Copyright (C) 2017 Alpha Griffin
# @%@~LICENSE~@%@

"""
TF_Curses - Build Model
Alphagriffin.com
Eric Petersen @Ruckusist <eric.alphagriffin@gmail.com>
"""

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.1"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Prototype"

print("Alpha Griffin TF_Curses Project")

import tensorflow as tf
import argparse
import sys
import os
from time import sleep
import ag.logging as log


host = '' # os . gethost
port_s = '2222'
# port_w = '2223'
name = 'AlphaGriffin_TF_Server'

cluster = tf.train.ClusterSpec({name: ["genruckus:2223", 
										  "agserver:2223"]})
									
def main():
	print("this server is running")
	server = tf.train.Server(cluster, job_name=name, task_index=0)
	while True:	  
		sleep(.5)
	
if __name__ == '__main__':
    try:
        main()
    except:
        log.error("and thats okay too.")
        sys.exit()
