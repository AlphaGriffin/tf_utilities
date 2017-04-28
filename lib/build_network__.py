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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import ag.logging as log
import tensorflow as tf
import numpy as np


working_path = os.getcwd()
class Options(object):
    def __init__(self):
        pass

class BuildModel(object):
    def __init__(self, conv, fc, outputs, options=None):
        self.num_conv = conv
        self.num_fc = fc
        self.num_outputs = outputs
        class Ops(): pass
        self.ops = Ops()

    def build_conv_layers(self, x_image, layers, f_size=5):

        start_shape = x_image
        log.debug("Start shape= {}".format(start_shape))
        reducing_shape = 0
        last_num_f = 0
        log.info("Building {} Convolutional Layers".format(layers))
        for layer in range(layers):
            num_f = 16 * (layer + 1)
            log.debug("Current Num of Features= {}".format(num_f))
            if layer == 0:
                img = start_shape
                channel = x_image.shape[3]
            else:
                img = reducing_shape
                channel = last_num_f

            last_num_f = num_f
            log.debug("Start shape= {}".format(start_shape))
            debugs = "New Features = {}".format(channel)
            reducing_shape,  = self.new_conv_layer(input=img,
                                                       filter_size=int(f_size),
                                                       chan=int(channel),
                                                       num_filters=int(num_f)
                                                       )
            layer_name = "convLayer_%s" % layer
            log.debug("Finishing Layer {}\n\t{}".format(layer_name, debugs))
            with tf.name_scope(layer_name):
                with tf.name_scope("conv_weights"):
                    tf.add_to_collection('weights', w)
                    tf.summary.histogram("conv_weights_{}".format(layer), w)
                    setattr(self, "conv_weights_{}".format(layer), w)
                with tf.name_scope("conv_biases"):
                    tf.add_to_collection('biases', b)
                    tf.summary.histogram("conv_biases_{}".format(layer), b)
                    setattr(self, "conv_biases_{}".format(layer), b)
            log.info("Finished Layer {}\n\t{}".format(layer_name, debugs))

        log.info("Finished Building {} Conv Layers\nMoving To Flattening...".format(layers))
        return reducing_shape, last_num_f

    def build_fc_layers(self, x_image, num_fc, num_final, keep_prob):
        log.info("Building {} Fully Connected Layers".format(num_fc))
        # first time thru options
        inputs = 0
        features = 0
        use_reLu = True
        use_Drop = True
        current_layer = x_image
        for layer in range(num_fc):
            if layer == 0:  # if first time through
                layer_shape = x_image.get_shape()
                features = layer_shape[1:4].num_elements()
            # always set in to features and ...
            inputs = features
            # set features to a decay function of i
            features = int(inputs / int(layer + 1))  # avoid a 0 zero divion error..

            if layer == num_fc - 1:  # if last time through
                features = num_final
                use_reLu = False  # dont use on the last time thru
                use_Drop = False

            current_layer, w, b = self.new_fc_layer(input_=current_layer,
                                                    num_inputs=inputs,
                                                    num_outputs=features,
                                                    keep_prob=keep_prob,
                                                    use_relu=use_reLu,
                                                    use_drop=use_Drop
                                                    )

            layer_name = "fullyLayer_{}".format(layer)
            with tf.name_scope(layer_name):
                with tf.name_scope("fc_weights"):
                    tf.summary.histogram("fc_weights_{}".format(layer), w)
                    tf.add_to_collection('weights', w)
                    setattr(self, "fc_weights_{}".format(layer), w)
                with tf.name_scope("fc_biases"):
                    tf.summary.histogram("fc_biases_{}".format(layer), b)
                    tf.add_to_collection('biases', b)
                    setattr(self, "fc_biases_{}".format(layer), b)
        return current_layer  # final layer

    def new_fc_layer(self, input_, num_inputs, num_outputs, keep_prob, use_relu=True, use_drop=False):
        weights = self.new_weights(shape=[num_inputs, num_outputs])  # set weights
        biases = self.new_biases(length=num_outputs)  # set number of OUTPUTS like give me top k or whatever...
        layer = tf.matmul(input_, weights) + biases  # this is a #BIGMATH func
        if use_relu:
            layer = tf.nn.relu(layer)
        if use_drop:
            layer = tf.nn.dropout(layer, keep_prob)
        return layer, weights, biases

    def new_weights(self, shape):
        """This generates a new weight for each layer"""
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weight", collections=["weights"])

    def new_biases(self, length):
        """This generates a new bias for each layer"""
        return tf.Variable(tf.constant(0.1, shape=[length]), collections=["biases"])

    def new_conv_layer(self, input, filter_size, chan, num_filters, use_pooling=False):
        X_shape = [filter_size, filter_size, chan, num_filters]
        weights = self.new_weights(shape=X_shape)
        biases = self.new_biases(length=num_filters)
        """ THis is the MAGIC again... """
        layer = tf.nn.conv2d(input=input,           # This is the output of the last layer
                             filter=weights,        # this is a thing
                             strides=[1, 1, 1, 1],  # 1111 is NOT pooled MAX work
                             padding='SAME')        # input output transformation
        layer += biases  # Add biases
        if use_pooling:  # this skips pixels... saves time but skips things obviously
            log.debug("using pooling")
            layer = tf.nn.max_pool(value=layer,           # take the weights and bias together as an input
                                   ksize=[1, 2, 2, 1],    # stuff
                                   strides=[1, 2, 2, 1],  # 2 x 2 stride... could increase... check
                                   padding='SAME')        # i feel like this should already be in variable, but w/e
        layer = tf.nn.relu(layer)                         # rectified linear Unit ... like a boss
        log.debug("Finished Building a conv Layer:\n\t{}".format(layer))
        return layer, weights, biases

    def flatten_layer(self, layer, num_features):
        log.info("Tranitioning from Conv to fc layers... {}".format(num_features))
        #layer_shape = layer.get_shape()  # ASSERT layer_shape == [num_images, img_height, img_width, num_channels]
        #num_features = layer_shape[1:4].num_elements()  # like a boss...
        num_features = num_features * 2 # do something cooler here... think gaussianly...
        # log.debug("Shape: {}, Features: {}".format(layer_shape, num_features))
        layer_flat = tf.reshape(layer, [-1, num_features])  # yep...
        log.info("Finished Flattening Layers")
        return layer_flat, num_features

    def build_inputs(self, dataset_h, dataset_w, dataset_c, dataset_classes):
        with tf.variable_scope('inputs'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            learn_rate = tf.train.exponential_decay(0.1,
                                                    global_step,
                                                    .000005,
                                                    0.87,
                                                    staircase=True,
                                                    name="Learn_decay")
            tf.summary.histogram('decay_rate', learn_rate)
            Input_Tensor_Image = tf.placeholder(tf.float32, [None,
                                                             dataset_h,
                                                             dataset_w,
                                                             dataset_c])
            Input_Tensor_Labels = tf.placeholder(tf.float32, [None,
                                                              dataset_classes])
            keep_prob = tf.placeholder(tf.float32, name="Keep_prob")
        tf.add_to_collection('global_step', global_step)
        tf.add_to_collection('learn_rate', learn_rate)
        tf.add_to_collection('input_tensor', Input_Tensor_Image)
        tf.add_to_collection('label_tensor', Input_Tensor_Labels)
        tf.add_to_collection('keep_prob', keep_prob)
        return Input_Tensor_Image, Input_Tensor_Labels, learn_rate, keep_prob

    def build_outputs(self, x_image, num_conv, num_fc, num_outputs, keep_prob):
        with tf.variable_scope("conv_layers"):
            x_image, num_features = self.build_conv_layers(x_image, num_conv)
        with tf.variable_scope("Flat_layer"):
            x_image, features = self.flatten_layer(x_image, num_features)
        with tf.variable_scope("FC_layers"):
            final_layer = self.build_fc_layers(x_image,
                                               num_fc,
                                               num_final=num_outputs,
                                               keep_prob=keep_prob
                                               )
        return final_layer

    def training_method(self, x_image, input_label, learn_rate):
        print("new test")
        with tf.variable_scope("mupen_method"):
            cost = tf.reduce_mean(tf.square(tf.subtract(input_label, x_image)))
            train_vars = tf.trainable_variables()
            loss = cost + \
                   tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * \
                   0.001
            train = tf.train.AdamOptimizer(1e-5).minimize(loss)

        tf.add_to_collection("loss", loss)
        tf.summary.scalar("loss", loss)
        tf.add_to_collection("train", train)
        return train, loss, cost



def main():
    # hoping to use an ini file here... this class will probably still parse that though
    options = Options()
    # TODO: this is still missing a command line arg parser!
    app = TF_Curses(options)
    try:
        app.main()
        os.system('clear')
    except KeyboardInterrupt:
        app.exit_safely()
        os.system('clear')
        pass

if __name__ == '__main__':
    try:
        main()
    except:
        log.error("and thats okay too.")
