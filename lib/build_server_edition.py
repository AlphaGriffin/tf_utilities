# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# !/usr/bin/env python2.7
"""Train and export a simple Softmax Regression TensorFlow model.

The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.

Usage: mnist_export.py [--training_iteration=x] [--model_version=y] export_dir
"""

import os
import sys

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

from tensorflow_serving.example import mnist_input_data

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


def main(_):
    if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
        print('Usage: mnist_export.py [--training_iteration=x] '
              '[--model_version=y] export_dir')
        sys.exit(-1)
    if FLAGS.training_iteration <= 0:
        print
        'Please specify a positive value for training iteration.'
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print
        'Please specify a positive value for version number.'
        sys.exit(-1)

    # Train model
    print
    'Training model...'
    mnist = mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)


    sess = tf.InteractiveSession()
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32), }

    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    global_step = tf.Variable(0, name='global_step', trainable=False, collections=[global_step])
    x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
    y_ = tf.placeholder('float', shape=[None, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    sess.run(tf.initialize_all_variables())
    y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy, global_step=global_step)
    values, indices = tf.nn.top_k(y, 10)
    prediction_classes = tf.contrib.lookup.index_to_string(
        tf.to_int64(indices), mapping=tf.constant([str(i) for i in xrange(10)]))
    for _ in range(FLAGS.training_iteration):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print
    'training accuracy %g' % sess.run(
        accuracy, feed_dict={x: mnist.test.images,
                             y_: mnist.test.labels})
    print
    'Done training!'

    # Export model
    # WARNING(break-tutorial-inline-code): The following code snippet is
    # in-lined in tutorials, please update tutorial documents accordingly
    # whenever code changes.
    export_path_base = sys.argv[-1]
    export_path = os.path.join(
        compat.as_bytes(export_path_base),
        compat.as_bytes(str(FLAGS.model_version)))
    print
    'Exporting trained model to', export_path
    builder = saved_model_builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    classification_inputs = utils.build_tensor_info(serialized_tf_example)
    classification_outputs_classes = utils.build_tensor_info(prediction_classes)
    classification_outputs_scores = utils.build_tensor_info(values)

    classification_signature = signature_def_utils.build_signature_def(
        inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
        outputs={
            signature_constants.CLASSIFY_OUTPUT_CLASSES:
                classification_outputs_classes,
            signature_constants.CLASSIFY_OUTPUT_SCORES:
                classification_outputs_scores
        },
        method_name=signature_constants.CLASSIFY_METHOD_NAME)

    tensor_info_x = utils.build_tensor_info(x)
    tensor_info_y = utils.build_tensor_info(y)

    prediction_signature = signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x},
        outputs={'scores': tensor_info_y},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.initialize_all_tables(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()

    print
    'Done exporting!'


if __name__ == '__main__':
    tf.app.run()





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
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
        """Start a Full Graph Scope"""
        with tf.variable_scope('Full_Graph'):
            """ Record Keeping """
            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                               trainable=False)
            with tf.name_scope("learn_rate"):
                self.learn_rate = tf.train.exponential_decay(
                    0.1, self.global_step,
                    1e5, 0.96, staircase=True,
                    name="Learn_decay")
            tf.summary.scalar("learn_rate", self.learn_rate)

            """ Do Basic Steps """
            with tf.name_scope("input"):
                self.Input_Tensor_Images = tf.placeholder(tf.float32, [None, self.dataset.height, self.dataset.width,
                                                                       self.dataset.num_channels], name="Input_Tensor")
                self.Input_Tensor_Labels = tf.placeholder(tf.float32, [None, self.dataset.num_classes],
                                                          name="Input_Label")
                self.Input_True_Labels = tf.argmax(self.Input_Tensor_Labels, dimension=1)
                self.x_image = self.Input_Tensor_Images  # current default layer

            with tf.name_scope("keep_prob"):
                self.keep_prob = tf.placeholder(tf.float32, name="Keep_prob")  # new feature goes with the dropout option

            """ Do Advanced Steps """
            with tf.name_scope("adv_steps"):
                self.convlayerNames, self.Conv_layers, self.Conv_weights = self.BUILD_CONV_LAYERS()  ## BUILD LAYERS
                self.x_image, self.features = self.flatten_layer(self.x_image)  ## SWITCH TO FC LAYERS
                self.fclayerNames, self.fc_layers = self.BUILD_FC_LAYER(self.options.fc_layers)  # build FC LAYERS
                # this is used but i dont know what to call it.

        with tf.name_scope("softmax"):
            self.Output_True_Layer = tf.nn.softmax(self.x_image, name="Final_Output")
        tf.summary.histogram('activations', self.Output_True_Layer)

        """ Variables """
        with tf.name_scope("Training_Methods"):
            with tf.name_scope("cross_entropy_softmax"):
                self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.x_image,
                                                                             labels=self.Input_Tensor_Labels)
                self.entropy_loss = tf.reduce_mean(self.cross_entropy)
            tf.summary.scalar("cross_entropy", self.entropy_loss)

            with tf.name_scope('Entropy_Optimizer_Train'):
                self.train_ent_loss = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.entropy_loss)

            with tf.name_scope('train'):
                self.cost = tf.reduce_mean(tf.square(tf.subtract(self.Input_Tensor_Labels, self.x_image)))
                training_vars = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 0.001
                self.loss = self.cost + training_vars
            tf.summary.scalar("train_cost", self.loss)
            with tf.name_scope('Dropout_Optimizer_Train'):
                 self.train_drop_loss = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss)
                # self.train_drop_loss = tf.train.AdagradOptimizer(learning_rate=self.learn_rate).minimize(self.loss)

        """ Finishing Steps """
        with tf.name_scope("accuracy"):
            with tf.name_scope('correct_prediction'):
                self.Output_True_Labels = tf.argmax(self.Output_True_Layer, dimension=1)
                self.correct_prediction = tf.equal(self.Output_True_Labels, self.Input_True_Labels)
            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        """This is some tricks to push our matplotlib graph inside tensorboard"""

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
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name="weight")

    def new_biases(self, length):
        """This generates a new bias for each layer"""
        return tf.Variable(tf.constant(0.05, shape=[length]), name="bias")

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