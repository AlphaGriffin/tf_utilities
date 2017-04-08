#!/usr/bin/python
# Copyright (C) 2016 Alpha Griffin
# Copyright (C) 2017 Alpha Griffin

"""
Objective
---------
    build a more dynamic and easily tweakable TF frontend for eventual GUI.

Progress
--------
    2-25-17: Adding more decoration to functions for long term readability
    and use. Also making ready for the first complete run of TKart.

    3-7-17: Going to start renaming some variables to bring them toward
    compliance with pep, god forbid im out of compliance. also we need to
    streamline some non-distributed known use cases.

TODO
----
    * write a demo on how to go from start to finish. do a youtube video,
      build a docker. build a win exe. build wtf mac uses.
    * do a lot of catch variables for setup use... be verbose.
    * finish tensorboard dev output progess step. - done
    * finish distrubted gpu progess step.  - mostly done.. supervisor is impelmented
    * finish UML output image.

Target for master push
----------------------
    * produce the tutorial results of the MNIST lesson and the TKart lesson
      with this setup.
    * produce a Tensorboard output with a guide for setup
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import skimage.io as io
from datetime import timedelta
# import numpy as np # SEEDS DAMNIT!
import time
from tqdm import tqdm
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# % matplotlib inline

"""!!!DEV BUILD IN PROGRESS!!!"""

class Network(object):
    def __init__(self, network):
        self.network = network
        pass

    def build_vars(self):
        pass

    def build_layers(self): pass

    def training_op(self, loss, lr):
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdagradOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False, collections=[global_step])
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


class Build_Adv_Network(object):
    """
    Dynamically create a Tensorflow Network from the given Input Options.
    """

    def __init__(self, dataset=None, init=True):
        """
        example
        -------
        >>> network = build_network.Build_Adv_Network(Mupen64_dataset)
        """
        self.dataset = dataset
        if dataset is not None:
            self.options = self.dataset.options
        else:
            self.options = 0
        # this is in the wrong place ... i think
        # self.step_size = int(self.dataset._num_examples / self.options.batch_size)

        # HERE WE GO!!

        # first start a new logDir folder DONT MESS THIS UP! ...
        # self.logDir()
        #if init: self.init_new_graph();

    def init_new_graph(self):
        self.build_default_values('/gpu:0')
        print("New Network is prepared for processing!")

    ###################################################################
    """This is bassically the init."""
    ###################################################################

    def build_default_values(self, worker=None):
        """
        This builds out the model from the options for a TF Graph

        Param
        -----
        worker: "/gpu:0"
            TODO:
            x = [y for y in worker.split(',')]
            if len(x) > 1:
                we have a distributed network
            else:
                we have a single computer
        """
        # with tf.device(tf.train.replica_device_setter(worker_device=worker, cluster=cluster)):
        # with tf.device(worker):

        """Start a Full Graph Scope"""
        with tf.name_scope('Full_Graph'):
            """ Record Keeping """
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                               trainable=False)
            with tf.variable_scope("learn_rate"):
                self.learn_rate = tf.train.exponential_decay(
                    0.1, self.global_step,
                    self.options.learning_rate, 0.87, staircase=True,
                    name="Learn_decay")
            tf.add_to_collection("learn_rate", self.learn_rate)
            tf.summary.scalar("learn_rate", self.learn_rate)

            """ Do Basic Steps """
            with tf.variable_scope("input"):
                self.Input_Tensor_Images = tf.placeholder(tf.float32, [None, self.dataset.height, self.dataset.width,
                                                                       self.dataset.num_channels], name="Input_Tensor")
                tf.add_to_collection("input", self.Input_Tensor_Images)
                self.Input_Tensor_Labels = tf.placeholder(tf.float32, [None, self.dataset.num_classes],
                                                          name="Input_Label")
                tf.add_to_collection("label", self.Input_Tensor_Labels)
                self.Input_True_Labels = tf.argmax(self.Input_Tensor_Labels, dimension=1)
                self.x_image = self.Input_Tensor_Images  # current default layer

            with tf.variable_scope("keep_prob"):
                self.keep_prob = tf.placeholder(tf.float32, name="Keep_prob")  # new feature goes with the dropout option
                tf.add_to_collection("keep_prob", self.keep_prob)
            """ Do Advanced Steps """
            with tf.variable_scope("adv_steps"):
                self.convlayerNames, self.Conv_layers, self.Conv_weights = self.BUILD_CONV_LAYERS()  ## BUILD LAYERS
                self.x_image, self.features = self.flatten_layer(self.x_image)  ## SWITCH TO FC LAYERS
                self.fclayerNames, self.fc_layers = self.BUILD_FC_LAYER(self.options.fc_layers)  # build FC LAYERS
                # this is used but i dont know what to call it.

        with tf.variable_scope("softmax"):
            self.Output_True_Layer = tf.nn.softmax(self.x_image, name="Final_Output")
            tf.add_to_collection("final_layer", self.Output_True_Layer)
        tf.summary.histogram('activations', self.Output_True_Layer)

        """ Variables """
        with tf.name_scope("Training_Methods"):
            with tf.variable_scope("cross_entropy_softmax"):
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.x_image,
                                                                             labels=self.Input_Tensor_Labels)
                self.entropy_loss = tf.reduce_mean(self.cross_entropy)
            tf.summary.scalar("cross_entropy", self.entropy_loss)

            with tf.variable_scope('Entropy_Optimizer_Train'):
                self.train_ent_loss = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.entropy_loss, global_step=self.global_step)

            with tf.variable_scope('train'):
                self.cost = tf.reduce_mean(tf.square(tf.subtract(self.Input_Tensor_Labels, self.x_image)))
                training_vars = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 0.001
                self.loss = self.cost + training_vars
                tf.add_to_collection("loss", self.loss)
            tf.summary.scalar("train_cost", self.loss)
            with tf.variable_scope('Dropout_Optimizer_Train'):
                 self.train_drop_loss = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
                 tf.add_to_collection("train_op", self.train_drop_loss)
                # self.train_drop_loss = tf.train.AdagradOptimizer(learning_rate=self.learn_rate).minimize(self.loss)

        """ Finishing Steps """
        with tf.variable_scope("accuracy"):
            with tf.variable_scope('correct_prediction'):
                self.Output_True_Labels = tf.argmax(self.Output_True_Layer, dimension=1)
                self.correct_prediction = tf.equal(self.Output_True_Labels, self.Input_True_Labels)
            with tf.variable_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        """This is some tricks to push our matplotlib graph inside tensorboard"""
        # with tf.variable_scope('Matplotlib_Input'):
        # Matplotlib will give us the image as a string ...
        #    self.matplotlib_img = tf.placeholder(dtype=tf.string.real_dtype, shape=[])
        # ... encoded in the PNG format ...
        #    my_img = tf.image.decode_png(self.matplotlib_img, 4)
        # ... that we transform into an image summary
        #    self.img_summary = tf.summary.image(
        #        'matplotlib_graph'
        #        , tf.expand_dims(my_img, 0)
        #    )


        """ Initialize the session """
        self.init_op = tf.global_variables_initializer()

        """create summary op"""
        self.merged = tf.summary.merge_all()

        """ Create Saver object"""
        self.saver = tf.train.Saver(
            #var_list={"{}".format(v): v for v in [tf.model_variables()]},
            write_version=tf.train.SaverDef.V2,
            sharded=True,
            keep_checkpoint_every_n_hours=.001
        )

        """ Create Supervisor Object"""
        """
        self.bossMan = tf.train.Supervisor(is_chief=True,
                                           logdir=self.options.logDir,
                                           checkpoint_basename='alpha.griffin',
                                            init_op =  self.init_op,
                                           summary_op=self.merged,
                                           saver=self.saver,
                                           global_step=self.global_step,
                                           save_model_secs=60)
        print("We are passing this test.")
        """
    ###################################################################
    """These are all impelemented in the above build function"""
    ###################################################################
    """
    def build_plotter(self, sess, input_images,):
        #setup matplotlib output for tensorboard
        inputs = np.array([ [(i - 1000) / 100] for i in input_images ])
        y_true_res, y_res = sess.run([y_true, y], feed_dict={ x: inputs
        })
        # We plot it using matplotlib
        # (This is some matplotlib wizardry to get an image as a string,
        # read the matplotlib documentation for more information)
        plt.figure(1)
        plt.subplot(211)
        plt.plot(inputs, y_true_res.flatten())
        plt.subplot(212)
        plt.plot(inputs, y_res)
        imgdata = io.BytesIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        # We push our graph into TensorBoard
        plot_img_summary = sess.run(img_summary, feed_dict={
            img_strbuf_plh: imgdata.getvalue()
        })
        sw.add_summary(plot_img_summary, i + 1)
        plt.clf()


    #depricated for use in tf.Supervisor
    def save_graph(self, session, path=None):
        #Manually save the graph
        saver = self.saver
        if path is None:
            path = self.options.save_path + 'StupidAbritraryFileName'
        saver.save(sess=session, save_path=path)
    """

    def BUILD_CONV_LAYERS(self):
        """
        Magically construct many Convultional layers for a Neural Network.
        :return:
            An unused list of all the layers_names, weights and biases for each layer. these are added to tensorboard
            automatically.
        """
        layers = self.options.conv_layers
        self.conv_layers_nameslist = []
        self.conv_layers_list = []
        self.conv_layers_wlist = []
        self.conv_layers_blist = []
        start_shape = self.x_image
        reducing_shape = 0
        last_num_f = 0
        if self.options.verbose: print("Building Conv Layers");
        for layer in range(layers):
            f_size = self.options.f_size
            num_f = 16 * (layer + 1)
            if layer == 0:
                img = start_shape
                channel = self.dataset.num_channels
            elif layer > 0:
                img = reducing_shape
                channel = last_num_f
                if self.options.verbose: print("LAST NUMBER OF FILERS = INPUT CHANNELS =  %s" % channel)

            last_num_f = num_f
            if self.options.verbose: print("NEW NUMBER OF Channels = %s" % channel)
            reducing_shape, w, b = self.new_conv_layer(input=img,
                                                       filter_size=f_size,
                                                       chan=channel,
                                                       num_filters=num_f)
            layer_name = "convLayer_%s" % layer
            with tf.name_scope(layer_name):
                with tf.name_scope("weights"):
                    tf.summary.histogram("weights", w)
                with tf.name_scope("biases"):
                    tf.summary.histogram('biases', b)
                #with tf.name_scope("logits"):
                #    tf.summary.histogram('pre_activations', reducing_shape)
            self.conv_layers_nameslist.append(layer_name)
            self.conv_layers_list.append(reducing_shape)
            self.conv_layers_wlist.append(w)
            self.conv_layers_blist.append(b)
            self.x_image = reducing_shape
            if self.options.verbose: print("done with layer: %s" % layer)
            if self.options.verbose: print("#: Finished building %s:\n%s\n:##:\n" % (layer_name, self.x_image))

        if self.options.verbose: print("Finished Building %s Conv Layers" % layers);
        return self.conv_layers_nameslist, self.conv_layers_list, self.conv_layers_wlist

    def BUILD_FC_LAYER(self, layers):
        """
        Magically create a huge number of Fully connected layers.
        :param layers:
            Number of FC layers to create.
        :return:
            returns a human readable non used list of the layer names. These are added to tensorboard automatically.
        """
        self.fc_layers_nameslist = []
        self.fc_layers_list = []
        self.fc_layers_wlist = []
        self.fc_layers_blist = []
        if self.options.verbose: print("# input features size = %s" % self.features)
        # first time thru options
        input = self.features
        output = self.options.fc_size
        use_reLu = True
        use_Drop = True
        if self.options.verbose: print("Building Fully Connected Layers");
        for layer in range(layers):
            if layers == 0:  ## this is right ... just seems wrong becuase # of convs up... # of input fc goes down
                layer_shape = layer.get_shape()
                num_features = layer_shape[1:4].num_elements()
                self.features = num_features
            if layer > 0:  # if not first time through
                input = output
                output = int(self.options.fc_size / (layer + 1))

            if layer == layers - 1:  # if last time through
                output = self.dataset.num_classes
                use_reLu = False  # dont use on the last time thru
                use_Drop = False

            if self.options.verbose: print(" input layers: %s" % input)
            if self.options.verbose: print(" output layers: %s" % output)
            self.x_image, w, b = self.new_fc_layer(input=self.x_image,
                                                   num_inputs=input,
                                                   num_outputs=output,
                                                   use_relu=use_reLu,
                                                   use_drop=use_Drop)
            layer_name = "fullyLayer_%s" % layer
            with tf.name_scope(layer_name):
                with tf.name_scope("weights"):
                    tf.summary.histogram("weights", w)
                with tf.name_scope("biases"):
                    tf.summary.histogram('biases', b)
                # with tf.name_scope("logits"):
                    # tf.summary.histogram('pre_activations', self.x_image)
            self.fc_layers_nameslist.append(layer_name)
            self.fc_layers_list.append(self.x_image)
            self.fc_layers_wlist.append(w)
            self.fc_layers_blist.append(b)
            if self.options.verbose: print("building %s:\n%s" % (layer_name, self.x_image))
        if self.options.verbose: print("Finished Building %s Fully Connected Layers" % layers);
        return self.fc_layers_nameslist, self.fc_layers_list

    def new_weights(self, shape):
        """This generates a new weight for each layer"""
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weight")

    def new_biases(self, length):
        """This generates a new bias for each layer"""
        return tf.Variable(tf.constant(0.1, shape=[length]), name="bias")

    def new_conv_layer(self, input, filter_size, chan, num_filters, use_pooling=True):
        """
        :param input:
        :param filter_size:
        :param chan:
        :param num_filters:
        :param use_pooling:
        :return:
            This layer, weights and biases for tensorboard output
        """
        if self.options.verbose: print("#:\tStarting new conv Layer!...")
        X_shape = [filter_size, filter_size, chan, num_filters]
        weights = self.new_weights(shape=X_shape)
        if self.options.verbose: print("#: weights shape = %s" % X_shape)
        biases = self.new_biases(length=num_filters)

        """ THis is the MAGIC again... """
        layer = tf.nn.conv2d(input=input,  # This is the output of the last layer
                             filter=weights,  # this is a thing
                             strides=[1, 1, 1, 1],  # 1111 is NOT pooled MAX work
                             padding='SAME')  # input output transformation
        layer += biases  # Add biases
        if use_pooling:  # this skips pixels... saves time but skips things obviously
            layer = tf.nn.max_pool(value=layer,  # take the weights and bias together as an input
                                   ksize=[1, 2, 2, 1],  # stuff
                                   strides=[1, 2, 2, 1],  # 2 x 2 stride... could increase... check
                                   padding='SAME')  # i feel like this should already be in variable, but w/e
        layer = tf.nn.relu(layer)  # rectified linear Unit ... like a boss
        if self.options.verbose: print("Finished Building a conv Layer:\n\t%s" % layer)
        return layer, weights, biases

    def flatten_layer(self, layer):
        layer_shape = layer.get_shape()  # ASSERT layer_shape == [num_images, img_height, img_width, num_channels]
        num_features = layer_shape[1:4].num_elements()  # like a boss...
        layer_flat = tf.reshape(layer, [-1, num_features])  # yep...
        if self.options.verbose: print(layer_flat)
        print("## DummyScript.com")
        return layer_flat, num_features

    def new_fc_layer(self, input, num_inputs, num_outputs, use_relu=True, use_drop=False):
        weights = self.new_weights(shape=[num_inputs, num_outputs])  # set weights
        biases = self.new_biases(length=num_outputs)  # set number of OUTPUTS like give me top k or whatever...
        layer = tf.matmul(input, weights) + biases  # this is a #BIGMATH func
        if use_relu:
            layer = tf.nn.relu(layer)
        if use_drop:
            layer = tf.nn.dropout(layer, self.keep_prob)
        return layer, weights, biases

    ###################################################################
    """These are New ideas maybe not impelmented yet..."""
    ###################################################################
    # Our UA function
    def univAprox(self, x, hidden_dim=50):
        # The simple case is f: R -> R
        input_dim = 1
        output_dim = 1

        with tf.variable_scope('UniversalApproximator'):
            ua_w = tf.get_variable(
                name='ua_w',
                shape=[input_dim, hidden_dim],
                initializer=tf.random_normal_initializer(stddev=.1))
            ua_b = tf.get_variable(
                name='ua_b',
                shape=[hidden_dim],
                initializer=tf.constant_initializer(0.))
            z = tf.matmul(x, ua_w) + ua_b
            a = tf.nn.relu(z)  # we now have our hidden_dim activations

            ua_v = tf.get_variable(
                name='ua_v',
                shape=[hidden_dim, output_dim],
                initializer=tf.random_normal_initializer(stddev=.1))
            z = tf.matmul(a, ua_v)

        return z