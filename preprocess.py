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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import ag.logging as log
from PIL import Image
import numpy as np
import tensorflow as tf

# !/usr/bin/python3
"""
Ruckusist @ alphagriffin.com
"""
import numpy as np


### IM HAVING A SEED PROBLEM!!!

class mupenDataset(object):
    """
    This is a Tensorflow Input Data Class... most of this output are required
    field for using the Advanced ModelBuilder and Processor

    TODO:
    -----
    (1) redo this whole thing with more sensible human readable labels
    (2) make a kickass human readable printout for .rst and console

    """

    def __init__(self, options, imgs=False, labels=False):
        self.name = 'MUPEN64+'
        self.options = options
        self.imgs = imgs  # full path passed in
        self.labels = labels  # full path passed in
        self.img_size = None  # is not square
        self.height = 66
        self.width = 200
        self.num_channels = 3
        self.num_classes = 5
        self.batch_size = self.options.batch_size
        self.img_size_flat = self.width * self.height

        # Necessary Placeholders for working being done
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = 0

        # hacks
        self.trainer = self

        # startup
        if imgs:
            self.build_return()

    def build_return(self):
        """ This opens the files and does the label argmax for you"""
        # probably _ all stuffs _ is a bad name for stuff ... bud it shouldnt be used
        self._all_images_, self._all_labels_ = self.load(self.imgs, self.labels)

        # this is used for a bunch of stuff
        self._num_examples = self._all_images_.shape[0]

        # split up Alldata into some chunks we can use
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.split(self._all_images_,
                                                                                              self._all_labels_)
        self.train_cls = np.array([label.argmax() for label in self.train_labels])
        self.test_cls = np.array([label.argmax() for label in self.test_labels])

        # This is good to know things are working
        if self.options.verbose: print(
            'ALL: images.shape: %s labels.shape: %s' % (self._all_images_.shape, self._all_labels_.shape))
        if self.options.verbose: print(
            'TRAIN: images.shape: %s labels.shape: %s' % (self.train_images.shape, self.train_labels.shape))
        if self.options.verbose: print(
            'TEST: images.shape: %s labels.shape: %s' % (self.test_images.shape, self.test_labels.shape))

    def load(self, images, labels):
        """ Load 2 numpy objects as a set images, labels in: paths, out: np.arrays"""
        images = np.load(images)
        if self.options.verbose: print("loaded {} images".format(len(images)))
        labels = np.load(labels)
        if self.options.verbose: print("loaded {} labels".format(len(labels)))
        return images, labels

    def split(self, images, labels):
        """ Split the dataset in to different groups for many reasons"""
        # this needs a SEED !!! OMG !!!!
        size = len(images)
        if len(labels) < size: size = len(labels)
        train_size = int(0.8 * size)
        idx = np.random.permutation(size)

        idx_train = idx[0:train_size]
        idx_valid = idx[train_size:]

        train_images = images[idx_train, :]
        train_labels = labels[idx_train, :]

        test_images = images[idx_valid, :]
        test_labels = labels[idx_valid, :]

        return train_images, train_labels, test_images, test_labels

    def next_batch(self, batch_size, shuffle=False):
        """ Shuffle is off by default """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)  # should add some sort of seeding for verification
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._all_images_[start:end], self._all_labels_[start:end]

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def tf_record(self, imgs, labels):
        imgs = self._all_images_
        labels = self._all_labels_

        tfrecords_filename = 'mupen64plus.tfrecords'

        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

        for img, label in (imgs, labels):
            h = img.shape[0]
            w = img.shape[1]

            img_raw = img.tostring()
            annotation_raw = label.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': self._int64_feature(h),
                'width': self._int64_feature(w),
                'image_raw': self._bytes_feature(img_raw),
                'mask_raw': self._bytes_feature(annotation_raw)}))

            writer.write(example.SerializeToString())

        writer.close()

        for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
            img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
                                                                  reconstructed_pair)
            print(np.allclose(*img_pair_to_compare))
            print(np.allclose(*annotation_pair_to_compare))












