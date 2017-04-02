#!/usr/bin/env python
"""Mupen64 video dataset
"""
__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.1"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Prototype"

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from PIL import Image

# TODO: this should have an arg parser... stuff...

class mupenDataset(object):

    def __init__(self, paths=None):
        # inital Values
        self.set_height = 200
        self.set_width = 100
        self.set_bw = False

        # inital Operations
        if paths is not None:
            if os.path.isdir(paths[0]):
                print("Processing Raw data for Path(s):-{}".format("\n\t\t\t\t-".join(p for p in paths)))
                dataset_ = self.process_raw_data(paths)
                filename = self.tf_record_save(dataset_)
                # this could be heavy...
                del dataset_
                dataset = self.tf_record_load(filename)
                print("Dataset loaded {}".format(filename))
            else:
                print("Loading a Chuck of TF_records {}".format(paths))
                dataset = self.tf_record_load(paths[0])
            if self.build_values(dataset):
                print("Loaded Dataset properly from {}".format(paths))
        else:
            debugs = "This will not do"
            print(debugs)

    def build_values(self, dataset):
        print("Building Values from the dataset")
        self.imgs = dataset[0]
        print(len(self.imgs))
        self.labels = dataset[1]
        print(len(self.labels))
        # debugs = ""
        self.imgs_shape = self.imgs.shape
        self.labels_shape = self.labels.shape
        self.num_examples = self.imgs.shape[0]
        self.height = self.imgs.shape[1]
        self.width = self.imgs.shape[2]
        # self.channels = self.imgs.shape[3]
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.split(self.imgs,
                                                                                              self.labels)
        self.train_cls = np.array([label.argmax() for label in self.train_labels])
        self.test_cls = np.array([label.argmax() for label in self.test_labels])

        debugs += "# Images = \{ #{}, h: {}, w: {}}".format(self.num_examples,
                                                            self.height,
                                                            self.width)
        debugs += "# imgs shape: {} labels shape: {}".format(self.imgs_shape, self.labels_shape)
        # print(debugs)
        return True

    def process_raw_data(self, paths):
        dataset_x = []
        dataset_y = []
        for i in paths:
            print("# \tProcessing folder:\n\t\t-{}".format(i))
            imgs, labels = self.gamepadImageMatcher(i)
            dataset_y.append(labels)
            print("sync complete.")

            for image in imgs:
                filename = os.path.join(i, image)
                img = self.prepare_image(filename)
                dataset_x.append(img)
        dataset_x = np.asarray(dataset_x)
        dataset_y = np.concatenate(dataset_y)
        print("Dataset Successfully Processed to numpy data")
        return dataset_x, dataset_y

    def make_BW(self, img):
        return np.dot(img[..., :3, 1], [0.299, 0.587, 0.114])

    def prepare_image(self, img):
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
        pil_image = Image.open(img)  # open img
        x = pil_image.resize((self.set_height, self.set_width), Image.ANTIALIAS)  # resizes image
        numpy_img = np.array(x)  # convert to numpy
        if self.set_bw:
            numpy_img = self.make_BW(numpy_img)  # grayscale
        return numpy_img

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
            # print ("CSV HAS NO DATA")
            return None, None

        # Get list of images in directory and sort it
        all_files = os.listdir(path)
        images = []
        for filename in all_files:
            if filename.endswith('.png'):
                images.append(filename)
        images = sorted(images)

        if not images:
            # print ("FOUND NO IMAGES");
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
            hyphen = imgfile.rfind('-')  # Get last index of '-'
            if hyphen < 0:
                break
            imgtime = int(imgfile[hyphen + 1:-4])  # cut it out!
            lastKeptWasImage = False  # Did we last keep an image, or a line?
            if imgtime > prev_csvtime:
                keep_images.append(imgfile)
                del images[0]
                lastKeptWasImage = True

                # We just kept an image, so we need to keep a
                # corresponding input row too
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
        return keep_images, keep_csv

    def split(self, images, labels):
        """ Split the dataset in to different groups for many reasons"""
        # this needs a SEED !!! OMG !!!!
        size = images.shape[0]
        print("initial split size: {}".format(size))
        if labels.shape[0] < size: size = labels.shape[0]
        train_size = int(0.8 * int(size))
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
                perm = np.arange(self._num_examples) # should add some sort of seeding for verification
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

    def tf_record_save(self, dataset_):
        print("Saving TF records")
        imgs = dataset_[0]
        labels = dataset_[1]

        print("{}".imgs[2])
        print("{}".labels[2])
        sys.exit()
        tfrecords_filename = "mupen64plus.tfrecords"
        save_path = os.path.join(os.getcwd(), "dataset", tfrecords_filename)
        debugs = "imgs: {}\nLabels: {}\nFileName: {}".format(imgs.shape,labels.shape,save_path)
        print(debugs)
        # sys.exit()
        with tf.python_io.TFRecordWriter(save_path) as writer:
            for img, label in zip(imgs, labels):
                h = img.shape[0]
                w = img.shape[1]
                c = img.shape[2]
                l = label.shape[0]
                img_raw = img.tostring()
                annotation_raw = label.tostring()
                # debugs += "\nh: {}, w: {}, c: {}, l:{}\r".format(h, w, c, l)
                # print(debugs)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': self._int64_feature(h),
                    'width': self._int64_feature(w),
                    'channels': self._int64_feature(c),
                    'actions': self._int64_feature(l),
                    'image_data': self._bytes_feature(img_raw),
                    'label_data': self._bytes_feature(annotation_raw)}))

                writer.write(example.SerializeToString())
        print("Saved Records were created.\n{}".format(debugs))
        return save_path

    def tf_record_load(self, tf_records):
        print("loading data")
        _images = []
        _labels = []
        record_iterator = tf.python_io.tf_record_iterator(path=tf_records)
        print("started record iterator")
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            height = int(example.features.feature['height']
                         .int64_list
                         .value[0])

            width = int(example.features.feature['width']
                        .int64_list
                        .value[0])

            channels = int(example.features.feature['channels']
                        .int64_list
                        .value[0])

            actions = int(example.features.feature['actions']
                        .int64_list
                        .value[0])

            img_string = (example.features.feature['image_data']
                          .bytes_list
                          .value[0])

            annotation_string = (example.features.feature['label_data']
                                 .bytes_list
                                 .value[0])
            # print("h: {}, w: {}, c: {}, l: {} ".format(height, width, channels, actions))
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            np_image = np.reshape(img_1d, (height, width, channels))

            #image_shape = tf.stack([height, width, channels])
            #reconstructed_img = tf.reshape(img_1d, image_shape)
            #print("{}".format(reconstructed_img.shape))
            annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
            np_label = np.reshape(annotation_1d, actions)
            # Annotations don't have depth (3rd dimension)
            # reconstructed_annotation = annotation_1d.reshape((actions, -1))

            _images.append(np_image)
            _labels.append(np_label)

        ## still need to reshape for use...
        #images = np.reshape(reconstructed_images[0], reconstructed_images[0][0].shape)
        #labels = reconstructed_images[1]
        #labels_shape = tf.stack([labels[0]])
        #labels_reshape = np.reshape(labels[0], labels[1])

        print("Label example = {}".format(_labels[55]))

        #print("Images Shape: {}".format(images.shape))
        # print("len of just the thing {}, thing sub 1 {}, thing sub 2 {},3 {}, 4 {}".format(len(_images), len(_images[0]), len(_images[1]), len(_images[2]), len(_images[3])))
        print("finished Loading dataset... wow that was fast...")
        sys.exit()
        return reconstructed_images, reconstructed_annotation

    def read_and_decode(self, tf_records):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(tf_records)

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'channels': tf.FixedLenFeature([], tf.int64),
                'actions': tf.FixedLenFeature([], tf.int64),
                'image_data': tf.FixedLenFeature([], tf.string),
                'label_data': tf.FixedLenFeature([], tf.string)
            })

        images = tf.decode_raw(features['image_data'], tf.uint8)

        labels = tf.decode_raw(features['label_data'], tf.uint8)

        print("loading files -\n\t{}\n\t{}".format(images, labels))

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        channels = tf.cast(features['channels'], tf.int32)
        actions = tf.cast(features['actions'], tf.int32)
        print("Fount features:\n\th: {}, w: {}, c:{}, l:{}".format(height, width, channels, actions))


        image = tf.reshape(image, [width, height, channels])
        image = tf.cast(image, tf.float32)




        image_shape = tf.stack([height, width, channels])
        label_shape = tf.stack([actions, 1])
        print("Preshapeing: images: {}, labels: {}".format(image_shape, label_shape))
        images = tf.reshape(images, image_shape)
        labels = tf.reshape(labels, label_shape)
        print("Finished Loading TF_Records dataset... images {}, labels {}".format(images, labels))
        """
        image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
        annotation_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.int32)

        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.

        resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                               target_height=IMAGE_HEIGHT,
                                                               target_width=IMAGE_WIDTH)

        resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
                                                                    target_height=IMAGE_HEIGHT,
                                                                    target_width=IMAGE_WIDTH)

        images, annotations = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                     batch_size=2,
                                                     capacity=30,
                                                     num_threads=2,
                                                     min_after_dequeue=10)
        """
        return images, labels



