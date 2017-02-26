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
    
TODO
----
    * write a demo on how to go from start to finish. do a youtube video, 
      build a docker. build a win exe. build wtf mac uses.
    * do a lot of catch variables for setup use... be verbose.
    * finish tensorboard dev output progess step.
    * finish distrubted gpu progess step.
    * finish UML output image.
    
Target for master push
----------------------
    * produce the tutorial results of the MNIST lesson and the TKart lesson
      with this setup.
    * produce a Tensorboard output with a guide for setup
"""
import tensorflow as tf
from PIL import Image
import numpy as np

"""!!!DEV BUILD IN PROGRESS!!!"""

class Build_Adv_Network(object):
    """
    Dynamically create a Tensorflow Network from the given Input Options.
    """
    def __init__(self, dataset = None, init = True):
        self.dataset = dataset
        self.options = self.dataset.options
        # this is in the wrong place ... i think
        self.step_size = int(self.dataset._num_examples / self.options.batch_size)

        ## HERE WE GO!!
        if init: self.init_new_graph();
        else:
            self.init_load_graph();
        # IF NOT Pull up an old one
    """
    #def if model = none then show a list of available models in bank
    def save_graph(self):
        saver = tf.train.Saver()
        save_path = self.options.save_path + '_best_validation_1_'
        saver.save(sess=self.session, save_path=save_path)

    def load_graph(self, graph=None):
        if graph is None: graph = '_best_validation_1_'
        saver = tf.train.Saver()
        save_path = self.options.save_path + graph
        saver.restore(sess=self.session, save_path=save_path)
    
    TODO:
    def prepare_image(self, img):
        #print("DEBUGS  1 {}".format(img))
        pil_image = Image.open(img)# open img
        x = pil_image.resize((200, 66), Image.ANTIALIAS) # resizes image in-place
        numpy_img = np.array(x)         # convert to numpy
        #grey_numpy_image = self.make_BW(numpy_img)
        return numpy_img

    def que_graph(self, Image):
        try: vec = self.prepare_image(Image)
        except: vec = Image
        joystick = self.y.eval(feed_dict={self.x: [vec], self.keep_prob: 1.0})[0]
        output = [
                int(joystick[0] * 80),
                int(joystick[1] * 80),
                int(round(joystick[2])),
                int(round(joystick[3])),
                int(round(joystick[4])),
            ]

        for i in joystick:
            print("joystick: {}".format(i))
        print(output)
        return output

    def init_load_graph(self):
        self.session = tf.InteractiveSession()
        self.load_graph(self.session)
    """
    def init_new_graph(self):
        self.session = tf.InteractiveSession()
        self.build_default_values('/gpu:0')
        #self.saver = tf.train.Saver()

        if self.options.verbose: print("New Network is prepared for processing!");
    """
    TODO:
    def build_out_distrubited_graph(self):
        cluster = ""
        worker = ""
        server = ""
        
        # Calculate the learning rate schedule.
        num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 FLAGS.batch_size)
        
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
    
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        cifar10.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        
        gradient_cluster = []
        with tf.variable_scope(tf.get_variable_scope()):
            self.build_default_values()
        grads = average_gradients(tower_grads)
    """
    
    def build_default_values(self,worker):
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
        #with tf.device(tf.train.replica_device_setter(worker_device=worker, cluster=cluster)):
        with tf.device(worker):
            """ Record Keeping """
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            
            """ Do Basic Steps """
            self.Input_Tensor_Images = tf.placeholder(tf.float32, [None, self.dataset.height, self.dataset.width, self.dataset.num_channels], name="Input_Tensor")
            self.Input_Tensor_Labels = tf.placeholder(tf.float32, [None, self.dataset.num_classes], name="Input_Label")
            self.Input_True_Labels   = tf.argmax(self.Input_Tensor_Labels, dimension=1)
            self.x_image             = self.Input_Tensor_Images # current default layer
            self.keep_prob           = tf.placeholder(tf.float32) # new feature goes with the dropout option
    
            """ Do Advanced Steps """
            self.convlayerNames, self.Conv_layers, self.Conv_weights     = self.BUILD_CONV_LAYERS() ## BUILD LAYERS
            self.x_image, self.features                                  = self.flatten_layer(self.x_image) ## SWITCH TO FC LAYERS
            self.fclayerNames, self.fc_layers                            = self.BUILD_FC_LAYER(self.options.fc_layers) # build FC LAYERS
            train_vars                                                   = tf.trainable_variables()
            self.Output_True_Layer = tf.nn.softmax(self.x_image, name="Final Output")
            tf.summary.histogram('activations', self.Output_True_Layer)
    
            """ Working Maths """
            with tf.name_scope("cross_entropy"):
                self.cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(logits=self.x_image, labels=self.Input_Tensor_Labels)
                with tf.name_scope("total"):
                    self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy)
            tf.summary.scalar("cross_entropy", self.cross_entropy_mean)
    
            with tf.name_scope("train_loss"):
                self.train_loss = tf.square(tf.sub(self.Input_Tensor_Labels, self.x_image))
                self.cost       = tf.reduce_mean(self.train_loss)
                tf.summary.scalar("train_cost", self.cost)
                training_vars       = tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * self.options.L2NormConst
                self.loss           = self.cost + training_vars
            tf.summary.scalar("train_loss", self.loss)
    
            with tf.name_scope("train_ops"):
                self.optimizer      = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate).minimize(self.loss)
                self.optimizer2     = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate).minimize(self.cross_entropy_mean)
                self.train_op_3 = tf.train.AdagradOptimizer(self.options.learning_rate).minimize(
                                        self.loss, global_step=self.global_step)
                self.train_op_4 = tf.train.AdamOptimizer(self.options.learning_rate).minimize(
                                        self.loss, global_step=self.global_step)
    
            """ Finishing Steps """
            with tf.name_scope("accuracy"):
                with tf.name_scope('correct_prediction'):
                    self.Output_True_Labels = tf.argmax(self.Output_True_Layer, dimension=1)
                    self.correct_prediction = tf.equal(self.Output_True_Labels, self.Input_True_Labels)
                with tf.name_scope('accuracy'):
                    self.accuracy           = tf.reduce_mean(tf.cast (self.correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
    
            """ Initialize the session """
            init_op = tf.global_variables_initializer()
            
            """create summary op"""
            merged = tf.summary.merge_all()
    
            """ Create Saver object"""
            saver = tf.train.Saver(
                           var_list = {v.op.name: v for v in [train_vars,
                                                          self.conv_layers_wlist,
                                                          self.conv_layers_blist,
                                                          self.fc_layers_wlist,
                                                          self.fc_layers_blist,
                                                              ]},
                           write_version=tf.train.SaverDef.V2,
                           sharded=True,
                           keep_checkpoint_every_n_hours=1.0,
                           pad_step_number = False,
                           )
            
            """ Create Supervisor Object"""
            self.bossMan = tf.train.Supervisor(is_chief=True,
                                     logdir=self.options.logdir,
                                     init_op = init_op,
                                     summary_op = merged,
                                     saver = saver,
                                     global_step = global_step,
                                     save_model_secs = 600)
        

        
        #self.train_writer = tf.summary.FileWriter(summaries_dir + '/train',
        #                               sess.graph)
        #self.test_writer = tf.summary.FileWriter(summaries_dir + '/validation')

        #self.long_haul()
        
    def run_network(self, iters=5): pass
        #"""This will optimize the function"""
        
        
    """
    def long_haul(self):
        index = 0
        #step_size = int(self.dataset._num_examples / self.options.batch_size)
        for epoch in range(self.options.epochs):
            for i in range(63):
                batch = self.dataset.next_batch(self.options.batch_size)
                if len(batch[0]) is len(batch[1]):
                    self.optimize(batch,0.8)

                    if i % 100 == 0:
                      loss_value = self.get_lost(batch[0],batch[1])
                      print("epoch: {0} step: {1} loss: {2:.3}".format(epoch, index, loss_value))
                index += 1

    def optimize(self,batch, keep_prob):
        self.session.run(self.optimizer, feed_dict={self.Input_Tensor_Images: batch[0], self.Input_Tensor_Labels: batch[1], self.keep_prob: keep_prob})
        return True
        #train_step.run(feed_dict={self.x: batch[0], self.Y: batch[1], self.keep_prob: 0.8}) #! Hardcoded value

    def get_lost(self, x, y):
        loss_value = self.loss.eval(session=self.session,feed_dict={self.Input_Tensor_Images: x, self.Input_Tensor_Labels: y, self.keep_prob: 1.0})
        return loss_value #! hardcoded value
    """
    def BUILD_CONV_LAYERS(self):
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
            num_f = 16 * (layer+1)
            if layer == 0:
                img = start_shape
                channel = self.dataset.num_channels
            elif layer > 0:
                img = reducing_shape
                channel = last_num_f
                if self.options.verbose:print("LAST NUMBER OF FILERS = INPUT CHANNELS =  %s" % channel)

            last_num_f = num_f
            if self.options.verbose:print("NEW NUMBER OF Channels = %s" % channel)
            reducing_shape, w, b = self.new_conv_layer(input       = img,
                                                    filter_size = f_size,
                                                    chan        = channel,
                                                    num_filters = num_f)
            layer_name = "convLayer_%s" % layer
            with tf.name_scope(layer_name):
                with tf.name_scope("weights"):
                    self.summaries(w)
                with tf.name_scope("biases"):
                    self.summaries(b)
                with tf.name_scope("logits"):
                    tf.summary.histogram('pre_activations', reducing_shape)
            self.conv_layers_nameslist.append(layer_name)
            self.conv_layers_list.append(reducing_shape)
            self.conv_layers_wlist.append(w)
            self.conv_layers_blist.append(b)
            self.x_image = reducing_shape
            if self.options.verbose:print("done with layer: %s"%layer)
            if self.options.verbose: print("#: Finished building %s:\n%s\n:##:\n" % (layer_name, self.LAYER))

        if self.options.verbose: print("Finished Building %s Conv Layers" % layers);
        return self.conv_layers_nameslist, self.conv_layers_list, self.conv_layers_wlist

    def BUILD_FC_LAYER(self, layers):
        self.fc_layers_nameslist = []
        self.fc_layers_list = []
        self.fc_layers_wlist = []
        self.fc_layers_blist = []
        if self.options.verbose:print("# input features size = %s" % self.features)
        #first time thru options
        input = self.features
        output = self.options.fc_size
        use_reLu  = True
        use_Drop = True
        if self.options.verbose: print("Building Fully Connected Layers");
        for layer in range(layers):
            if layers == 0:  ## this is right ... just seems wrong becuase # of convs up... # of input fc goes down
                layer_shape = layer.get_shape()
                num_features = layer_shape[1:4].num_elements()
                self.features = num_features
            if layer > 0:         # if not first time through
                input = output
                output = int(self.options.fc_size / (layer+1))

            if layer == layers-1: # if last time through
                output = self.dataset.num_classes
                use_reLu  = False # dont use on the last time thru
                use_Drop = False

            if self.options.verbose:print(" input layers: %s" % input)
            if self.options.verbose:print(" output layers: %s" % output)
            self.x_image, w, b = self.new_fc_layer(input        = self.x_image,
                                                   num_inputs   = input,
                                                   num_outputs  = output,
                                                   use_relu     = use_reLu,
                                                   use_drop     = use_Drop)
            layer_name = "fullyLayer_%s" % layer
            with tf.name_scope(layer_name):
                with tf.name_scope("weights"):
                    self.summaries(w)
                with tf.name_scope("biases"):
                    self.summaries(b)
                with tf.name_scope("logits"):
                    tf.summary.histogram('pre_activations', self.x_image)
            self.fc_layers_nameslist.append(layer_name)
            self.fc_layers_list.append(self.x_image)
            self.fc_layers_wlist.append(w)
            self.fc_layers_blist.append(b)
            if self.options.verbose:print("building %s:\n%s" % (layer_name, self.x_image))
        if self.options.verbose: print("Finished Building %s Fully Connected Layers" % layers);
        return self.fc_layers_nameslist, self.fc_layers_list


    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name="weight")

    def new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]), name="bias")

    def new_conv_layer(self, input, filter_size, chan, num_filters, use_pooling=True):
        if self.options.verbose:print("#:\tStarting new conv Layer!...")
        X_shape = [filter_size, filter_size, chan, num_filters]
        weights = self.new_weights(shape  = X_shape)
        if self.options.verbose:print("#: weights shape = %s" % X_shape)
        biases = self.new_biases(length   = num_filters)

        """ THis is the MAGIC again... """
        layer = tf.nn.conv2d(input=input,                   # This is the output of the last layer
                             filter=weights,                # this is a thing
                             strides=[1, 1, 1, 1],          # 1111 is NOT pooled MAX work
                             padding='SAME')                # input output transformation
        layer += biases                                     # Add biases
        if use_pooling:                                     # this skips pixels... saves time but skips things obviously
            layer = tf.nn.max_pool(value = layer,           # take the weights and bias together as an input
                                   ksize = [1, 2, 2, 1],    # stuff
                                   strides = [1, 2, 2, 1],  # 2 x 2 stride... could increase... check
                                   padding='SAME')          # i feel like this should already be in variable, but w/e
        layer = tf.nn.relu(layer)                           # rectified linear Unit ... like a boss
        if self.options.verbose:print("Finished Building a conv Layer:\n\t%s" % layer)
        return layer, weights, biases

    def flatten_layer(self, layer):
        layer_shape = layer.get_shape()                    # ASSERT layer_shape == [num_images, img_height, img_width, num_channels]
        num_features = layer_shape[1:4].num_elements()     # like a boss...
        layer_flat = tf.reshape(layer, [-1, num_features]) # yep...
        if self.options.verbose:print(layer_flat)
        print("## DummyScript.com")
        return layer_flat, num_features

    def new_fc_layer(self, input, num_inputs, num_outputs, use_relu=True,use_drop=False):
        weights = self.new_weights(shape=[num_inputs, num_outputs])        # set weights
        biases = self.new_biases(length=num_outputs)                       # set number of OUTPUTS like give me top k or whatever...
        layer = tf.matmul(input, weights) + biases                         # this is a #BIGMATH func
        if use_relu:
            layer = tf.nn.relu(layer)
        if use_drop:
            layer = tf.nn.dropout(layer,self.keep_prob)
        return layer, weights, biases


    """
    def feed_single(self, layer, image=None):
        if image is not None: img = image;
        else: img = self.test_IMAGE
        feed_dict = {self.Input_Tensor_Images: [img]}
        values = self.session.run(layer, feed_dict=feed_dict)
        return values, img

    def prob_dictionary(self, test=True, x_batch=None, y_true_batch=None, keep=1.0):
        in_tensor = self.x_image
        in_tensor_label = self.Input_Tensor_Labels
        keep_prob = self.keep_prob

        if not test:
            dataset_dictionary = {in_tensor:x_batch,
                                  in_tensor_label:y_true_batch,
                                  keep_prob:keep}
        else:
            dataset_dictionary = { in_tensor: self.test_batch[0],
                                  in_tensor_label:self.test_batch[1],
                                  keep_prob:keep }
        return dataset_dictionary

    def feed_dictionary(self, test=True, x_batch=None, y_true_batch=None):
        dataset = self.dataset
        #return { INPUT_TENSOR: INPUT_IMG, INPUT_TENSOR_LABEL:input_label, inuput_true_: Test_cls}
        in_tensor = self.x_image
        in_tensor_label = self.Input_Tensor_Labels
        in_true = self.Input_True_Labels
        if test is not True:
            dataset_dictionary = {in_tensor: x_batch,
                                  in_tensor_label: y_true_batch}
        else:
            dataset_dictionary = {in_tensor: self.test_batch[0],
                                  in_tensor_label: dataset.test_labels,
                                  in_true: dataset.test_cls}
        return dataset_dictionary

    def BATCH_VERIFY(self, input_tensor, labels, cls_true):
        batch_size = self.options.classify_batch_size
        num_images = len(input_tensor)
        cls_pred = np.zeros(shape=num_images, dtype=np.int)
        i = 0
        while i < num_images:
            j = min(i + batch_size, num_images) # j is remade frest every loop...
            feed_dict = {self.Input_Tensor_Images: input_tensor[i:j], self.Input_Tensor_Labels: labels[i:j]}
            cls_pred[i:j] = self.session.run(self.y_pred_cls, feed_dict=feed_dict)
            i = j
        correct = (cls_true == cls_pred)
        return correct, cls_pred
    """
