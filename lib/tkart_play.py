#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 00:24:11 2017

@author: eric
"""

import tensorflow as tf


class Play(object):
    def __init__(self,options):
        self.options = options
        self.save_path = self.options.save_dir + '_best_validation_1_'
    
    def load_graph(self, session):
        session = tf.Session
        saver = tf.train.Saver()
        save_path = self.options.save_dir + '_best_validation_1_'
        saver.restore(sess=session, save_path=save_path)
            
    def classify(self, Image):
        img = prepare_image(Image)
        joystick = _best_validation_1_
        output = [
                int(joystick[0] * 80),
                int(joystick[1] * 80),
                int(round(joystick[2])),
                int(round(joystick[3])),
                int(round(joystick[4])),
            ]
            