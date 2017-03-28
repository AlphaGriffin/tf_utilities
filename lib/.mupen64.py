#!/usr/bin/python3
"""
Ruckusist @ alphagriffin.com
"""
import os, sys
import numpy as np
import tensorflow as tf
from PIL import Image
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
    def __init__(self, options, records=None, paths=None, numpy=None):
        self.name                  = 'MUPEN64plus'
        self.options               = options # use this for paths
        self.record                = records # A TFrecords dataset
        self.paths                 = paths
        # depricated
        # self.imgs                  = imgs    # full path passed in
        # self.labels                = labels  # full path passed in
        # would be cooler if were square
        self.img_size              = None    # is not square

        # all this should be dynamic from readying the dataset
        self.height                = 66
        self.width                 = 200
        self.num_channels          = 3
        self.num_classes           = 5
        self.batch_size            = self.options.batch_size
        self.img_size_flat         = self.width * self.height
        
        # Necessary Placeholders for working being done
        self._epochs_completed     = 0
        self._index_in_epoch       = 0
        self._num_examples         = 0

        # startup
        if paths:
            self.prepare_dataset()
            self.build_return()
        elif records:
            self.tf_record_load()
            self.build_return()
        
    def build_return(self):
        """ This opens the files and does the label argmax for you"""
        # this is used for a bunch of stuff
        self._num_examples = self._all_images_.shape[0]
        
        # split up Alldata into some chunks we can use 
        self.train_images, self.train_labels, self.test_images, self.test_labels = self.split(self._all_images_, self._all_labels_)
        self.train_cls = np.array([label.argmax() for label in self.train_labels])
        self.test_cls = np.array([label.argmax() for label in self.test_labels])
        
        # This is good to know things are working
        if self.options.verbose: print ('ALL: images.shape: %s labels.shape: %s' % (self._all_images_.shape, self._all_labels_.shape))
        if self.options.verbose: print ('TRAIN: images.shape: %s labels.shape: %s' % (self.train_images.shape, self.train_labels.shape))
        if self.options.verbose: print ('TEST: images.shape: %s labels.shape: %s' % (self.test_images.shape, self.test_labels.shape)) 
        
    def load(self, images, labels):
        """ Load 2 numpy objects as a set images, labels in: paths, out: np.arrays"""
        images = np.load(images)
        if self.options.verbose: print ("loaded {} images".format(len(images)))
        labels = np.load(labels)
        if self.options.verbose: print ("loaded {} labels".format(len(labels)))
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

    def next_batch2(self, batch_size, shuffle=False, test=False):
        """
        Shuffle is off by default
        Test is off by default... switches which set to take batch from
        """
        # which set are we using??
        images = self.train_images
        labels = self.train_labels
        if test:
            images = self.test_images
            labels = self.test_labels

        # get our start postition
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples) # should add some sort of seeding for verification
                np.random.shuffle(perm)
                images = images[perm]
                labels = labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        # batch check
        # if len(images) is not len(labels):
        #    return False
        return images[start:end], labels[start:end], self._epochs_completed

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

    def prepare_dataset(self, folders=None):
        dataset_x = []
        dataset_y = []
        save = self.options.save_dataset
        if folders is None:
            folders = self.paths
        for i in folders:
            if os.path.isdir(i):
                labels, imgs = self.gamepadImageMatcher(i)
                dataset_y.append(labels)

                for image in imgs:
                    img = self.prepare_image(os.path.join(i,image))
                    dataset_x.append(img)
        # complete the transfer
        dataset_x = np.asarray(dataset_x)
        dataset_y = np.concatenate(dataset_y)
        # save the files off
        if save:
            print("got 2 datasets:\n\t- {} : {}".format(dataset_x.shape,dataset_y.shape))
            self.tf_record_save(dataset_x, dataset_y)
            print("saved")
        self._all_images_ = dataset_x
        self._all_labels_ = dataset_y

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def tf_record_save(self, imgs, labels):
        """Create TF_records save file.
        :param imgs:
            a 3d array of 2d images as a dataset
        :param labels:
        :return:
        """
        tfrecords_filename = 'mupen64plus.tfrecords'

        with tf.python_io.TFRecordWriter(tfrecords_filename) as writer:
            for img, label in zip(imgs, labels):
                h = img.shape[0]
                w = img.shape[1]
                c = img.shape[2]
                l = labels.shape[1]
                print("Verify save dimensions: h: {}, w: {}, l: {}".format(h,w,l))
                image_data = img.tostring()
                label_data = label.tostring()

                dataset = tf.train.Example(features=tf.train.Features(feature={
                    'height': self._int64_feature(h),
                    'width': self._int64_feature(w),
                    'channel': self._int64_feature(c),
                    'actions': self._int64_feature(l),
                    'image_data': self._bytes_feature(image_data),
                    'label_data': self._bytes_feature(label_data)
                        }
                    )
                )

                writer.write(dataset.SerializeToString())
        print("Saved {}".format(tfrecords_filename))

    def tf_record_load(self, path=None):
        if path is None:
            path = self.record
        print("Loading Dataset from path:\n\t- {}".format(path))
        reconstructed_dataset = []
        imgs = []
        labels = []
        record_iterator = tf.python_io.tf_record_iterator(path=path)
        for string_record in record_iterator:
            # start a new example for every piece of data?
            example = tf.train.Example()
            example.ParseFromString(string_record)

            height = int(example.features.feature['height']
                         .int64_list
                         .value[0])

            width = int(example.features.feature['width']
                        .int64_list
                        .value[0])

            channel = int(example.features.feature['channel']
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
            print("Example: h: {}, w: {}, c: {} -- l: {}".format(height, width, channel, actions))
            # 1 dim Image...
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            # 3d set of 2d images... h, w, %elements
            dataset_x = img_1d.reshape(-1, (width, height, channel))
            # 1 dim labels
            annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
            # labels are a 2d array of 1d ... labels
            dataset_y = annotation_1d.reshape((actions, -1))
            imgs.append(dataset_x)
            labels.append(dataset_y)
            #reconstructed_dataset.append((dataset_x, dataset_y))
        # self._all_images_ = reconstructed_dataset[0][1]
        # self._all_labels_ = reconstructed_dataset[1][1]
        print("imgs: {},\nlabels: {}".format(len(imgs), len(labels)))
        sys.exit()
        print("Finished Loading Dataset... x: {}, y: {}".format(self._all_images_.shape, self._all_labels_.shape))

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
        return keep_csv, keep_images

    def make_BW(self, rgb):
        """
        This is the "rec601 luma" algorithm to compute 8-bit greyscale
        Parameters
        ----------
            img : numpy img

        Returns
        -------
            img : numpy image

        Example
        -------
        >>> img = Prepare.make_BW(img)
        """
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def prepare_image(self, img, makeBW=False):
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
        >>> img = Prepare.prepare_image(img, makeBW=False)
        """
        pil_image = Image.open(img)  # open img
        x = pil_image.resize((200, 66), Image.ANTIALIAS)  # resizes image
        numpy_img = np.array(x)  # convert to numpy
        if makeBW:
            numpy_img = self.make_BW(numpy_img)  # grayscale
        return numpy_img