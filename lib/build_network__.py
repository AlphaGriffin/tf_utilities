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

import ag.logging as log
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
working_path = os.getcwd()


class BuildModel(object):
    def __init__(self, conv, fc, outputs):
        self.num_conv = conv
        self.num_fc = fc
        self.num_outputs = outputs

    def build_conv_layers(self, x_image, layers, f_size=5):
        """
        Magically construct many Convolutional layers for a Neural Network.
        :return:
            An unused list of all the layers_names, weights and biases for each layer. these are added to tensorboard
            automatically.
        """
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
            reducing_shape, w, b = self.new_conv_layer(input=img,
                                                      filter_size=f_size,
                                                      chan=channel,
                                                      num_filters=num_f)
            layer_name = "convLayer_%s" % layer
            log.debug("Finishing Layer {}\n\t{}".format(layer_name, debugs))
            # tf.summary.histogram("weights{}".format(layer), w)
            # tf.summary.histogram("biases{}".format(layer), b)
            log.info("Finished Layer {}\n\t{}".format(layer_name, debugs))

        log.info("Finished Building {} Conv Layers\nMoving To Flattening...".format(layers))
        return reducing_shape, last_num_f

    def build_fc_layers(self, x_image, num_fc, num_final):
        """
        Magically create a huge number of Fully connected layers.
        :param layers:
            Number of FC layers to create.
        :return:
            returns a human readable non used list of the layer names. These are added to tensorboard automatically.
        """
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
            features = int(inputs / (layer + 1))  # avoid a 0 zero divion error..

            if layer == num_fc - 1:  # if last time through
                features = num_final
                use_reLu = False  # dont use on the last time thru
                use_Drop = False

            current_layer, w, b = self.new_fc_layer(input_=current_layer,
                                                    num_inputs=inputs,
                                                    num_outputs=features,
                                                    use_relu=use_reLu,
                                                    use_drop=use_Drop)

            layer_name = "fullyLayer_{}".format(layer)
            with tf.name_scope(layer_name):
                with tf.name_scope("weights"):
                    tf.summary.histogram("weights", w)
                with tf.name_scope("biases"):
                    tf.summary.histogram('biases', b)
        return current_layer  # final layer

    def new_fc_layer(self, input_, num_inputs, num_outputs, use_relu=True, use_drop=False):
        weights = self.new_weights(shape=[num_inputs, num_outputs])  # set weights
        biases = self.new_biases(length=num_outputs)  # set number of OUTPUTS like give me top k or whatever...
        layer = tf.matmul(input_, weights) + biases  # this is a #BIGMATH func
        if use_relu:
            layer = tf.nn.relu(layer)
        if use_drop:
            layer = tf.nn.dropout(layer, self.keep_prob)
        return layer, weights, biases

    def new_weights(self, shape):
        """This generates a new weight for each layer"""
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weight", collections=["weights"])

    def new_biases(self, length):
        """This generates a new bias for each layer"""
        return tf.Variable(tf.constant(0.1, shape=[length]), name="bias", collections=["biases"])

    def new_conv_layer(self, input, filter_size, chan, num_filters, use_pooling=False):
        X_shape = [filter_size, filter_size, chan, num_filters]
        weights = self.new_weights(shape=X_shape)
        biases = self.new_biases(length=num_filters)
        """ THis is the MAGIC again... """
        layer = tf.nn.conv2d(input=input,  # This is the output of the last layer
                             filter=weights,  # this is a thing
                             strides=[1, 1, 1, 1],  # 1111 is NOT pooled MAX work
                             padding='SAME')  # input output transformation
        layer += biases  # Add biases
        if use_pooling:  # this skips pixels... saves time but skips things obviously
            log.debug("using pooling")
            layer = tf.nn.max_pool(value=layer,  # take the weights and bias together as an input
                                   ksize=[1, 2, 2, 1],  # stuff
                                   strides=[1, 2, 2, 1],  # 2 x 2 stride... could increase... check
                                   padding='SAME')  # i feel like this should already be in variable, but w/e
        layer = tf.nn.relu(layer)  # rectified linear Unit ... like a boss
        log.debug("Finished Building a conv Layer:\n\t{}".format(layer))
        return layer, weights, biases

    def flatten_layer(self, layer):
        log.info("Tranitioning from Conv to fc layers")
        layer_shape = layer.get_shape()  # ASSERT layer_shape == [num_images, img_height, img_width, num_channels]
        num_features = layer_shape[1:4].num_elements()  # like a boss...
        log.debug("Shape: {}, Features: {}".format(layer_shape, num_features))
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
                                                             dataset_c],
                                                name="Input_Tensor")
            Input_Tensor_Labels = tf.placeholder(tf.float32, [None,
                                                              dataset_classes],
                                                 name="Input_Label")
            keep_prob = tf.placeholder(tf.float32, name="Keep_prob")
        tf.add_to_collection('global_step', global_step)
        tf.add_to_collection('learn_rate', learn_rate)
        tf.add_to_collection('input_tensor', Input_Tensor_Image)
        tf.add_to_collection('label_tensor', Input_Tensor_Labels)
        tf.add_to_collection('keep_prob', keep_prob)
        return Input_Tensor_Image, Input_Tensor_Labels, learn_rate

    def build_outputs(self, x_image, num_conv, num_fc, num_outputs):
        with tf.variable_scope("conv_layers"):
            x_image, num_features = self.build_conv_layers(x_image, num_conv)
        with tf.variable_scope("Flat_layer"):
            x_image, features = self.flatten_layer(x_image)
        with tf.variable_scope("FC_layers"):
            final_layer = self.build_fc_layers(x_image, num_fc, num_final=num_outputs)
        return final_layer

    def training_method(self, x_image, input_tensor, learn_rate):
        with tf.variable_scope("sophmax"):
            sophmax = tf.nn.softmax(x_image, name="sophmax")
        tf.add_to_collection("sophmax_layer", sophmax)

        with tf.variable_scope("mupen_method"):
            cost = tf.reduce_mean(tf.square(tf.subtract(input_tensor, x_image)))
            training_vars = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 0.001
            loss = cost + training_vars
        tf.add_to_collection("loss", loss)

        with tf.variable_scope('Dropout_Optimizer_Train'):
            train_drop_loss = tf.train.AdamOptimizer(learn_rate).minimize(loss)
        tf.add_to_collection("train_op", train_drop_loss);


    """
    def find_devices(self, ):
        # do a known port scan for TF services
        # parse that for ip and port
        local_workers = []
        server = tf.train.Server.create_local_server()
        # do a scan for local gpu??? or tf_gpu defaults to that...
        # im trying to split weights and matmuls ...
        local_gpus = []

        cluster = tf.train.ClusterSpec({"ps": server,
                                        "worker": local_workers})

        # Create and start a server for the local task.
        server = tf.train.Server(cluster,
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task_index)

        return local_workers, local_gpus

    def build_graph(self, num_conv, num_fc, num_outputs):
        # check the enviorment...
        workers, gpus = find_devices()

        # start a new graph for each worker.
        if workers:
            num_workers = len(workers)
            cluster =
            pass
            for worker, i in enumerate(workers):
                current_context = tf.Graph()
                current_worker = "/job:worker/task:%d" % FLAGS.task_index
                with current_context.container('AlphaGriffin_{}'.format(i)):
                    with tf.device(tf.train.replica_device_setter(
                            worker_device=current_worker,
                            cluster=cluster)):
                        # build up variables
                        input_tensor, input_tensor, learn_rate = build_inputs()
                        # take the inputed variables and implement them
                        x_image = build_outputs(input_tensor,
                                                num_conv,
                                                num_fc,
                                                num_outputs)
                        # this is a training method that should be...
                        # over written by another new class
                        _ = training_method(x_image, input_tensor, learn_rate)
                        # generic setup calls
                        init_op = tf.global_variables_initializer
                        merged = tf.summary.merge_all
    """

