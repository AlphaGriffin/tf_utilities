#!/usr/bin/env python
"""Provides A GUI for specific Machine Learning Use-cases.

TF_Curses is a frontend for processing datasets into machine
learning models for use in predictive functions.
"""
__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.1"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Prototype"

# imports
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import lib.build_network as net
import lib.mupen64 as data
import lib.options as opts
import ag.logging as log
from PIL import Image


def prepare_image(img, makeBW=False):
    """ This resizes the image to a tensorflowish size """
    log.debug("prepare_image: {}".format(img))
    try:
        pil_image = Image.open(img)  # open img
        log.debug("pil_image: {}".format(pil_image))
        x = pil_image.resize((200, 66), Image.ANTIALIAS)  # resizes image
        log.debug("pil_image resized: {}".format(x))
    except Exception as e:
        log.fatal("Exception: {}".format(e))
        return False
    numpy_img = np.array(x)
    return numpy_img



def classify(img, path):
    with tf.Session() as sess:
        print("restored graph in {}".format(path))
        new_saver = tf.train.import_meta_graph(os.path.join(path, "Alpha.meta"))
        new_saver.restore(sess, os.path.join(path, "Alpha"))
        x = tf.get_collection_ref('input')[0]
        k = tf.get_collection_ref('keep_prob')[0]
        y = tf.get_collection_ref('final_layer')[0]
        img = prepare_image(img)
        feed_dict = {x: [img], k:1.0}
        classification = sess.run(y, feed_dict)
        return classification



os.system("clear")
print("Starting Evaluation")
img = "/home/eric/.local/share/mupen64plus/screenshot/test.png"
path = os.path.join(os.getcwd(), "models", "log_27")
label = classify(img, path)
print(label)
