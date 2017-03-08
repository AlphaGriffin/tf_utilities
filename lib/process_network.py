import tensorflow as tf
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

"""
DESCRIPTION:
    TF - Manager "codenamed: process_network"

CURRENT WORK:
    feedback redoubts

TODO:
    complete load / save / cue transfer
    add a tf.supervisor loop
"""


class procNet(object):
    """This is a better way... V.2"""

    def __init__(self, network):
        self.network = network
        self.dataset = network.dataset
        self.options = network.options

    def run_network(self, iters=50, keep_prob=0.8):
        """
        This is a Basic optimization loop exploration... single user

        Params
        ------
        iters : 50 (default(is super low))
            this should be set yuge and then the system will turn its self off
            but the options.timeout switch far before it reaches this point.

        keep_prob : 0.8
            This is used in the dropout layer and can be passed a lower number
            for slower learning(better)?.

        Return
        ------
        nada

        Example
        ------
        >>> _, loss_value = build_network.Build_Adv_Network.basic_loop(iters=1e7)

        Todo
        ----
        * Do an advanced loop with the tf.Supervisor able to back out and use its
          advanced functionality.
        * Do a more advanced loop with the distrubuted network
        
        Notes:
        ------
        * adding a set of images from 1 per batch to the tensorboard.
        logdir="{}".format(self.options.logDir)
        """
        # be civilized
        start = time.time()
        with self.network.bossMan.managed_session() as sess:
            #sumwriter = tf.summary.FileWriter(logdir=self.options.logDir, graph=tf.get_default_graph())
            # tf.train.write_graph(tf.get_default_graph(), self.options.logDir, 'graph.pbtxt')
            for i in tqdm(range(iters)):
                while not self.network.bossMan.stop():
                    batch = self.dataset.next_batch(self.options.batch_size)
                    print("{}{}".format(batch[0], batch[1]))
                    if len(batch[0]) is len(batch[1]):
                        feed_dict = {self.network.Input_Tensor_Images: batch[0],
                                     self.network.Input_Tensor_Labels: batch[1],
                                     self.network.keep_prob: 0.8}

                        _, step, summary = sess.run([self.network.train_drop_loss,
                                                     self.network.global_step,
                                                     self.network.merged], feed_dict)


class Process_Network(object):
    """ This class handles Opimization / Visualization for TF Models """

    def __init__(self, network):
        self.network = network
        self.dataset = network.dataset
        self.options = network.options
        # GONNA NEED SOME GLOBALS FOR WORKFLOW CONTROLS
        self.werk_done = 0
        self.best_score = 0
        self.last_score = 0

    def end(self):
        """ This is run at the end of a TF script session"""
        self.network.session.close()

    def new_deal(self, iters=50, keep_prob=0.8):
        with self.network.session as sess:
            step = 0
            for i in tqdm(range(iters)):
                # while not self.bossMan.stop():
                batch = self.dataset.next_batch(self.options.batch_size, shuffle=True)
                if len(batch[0]) is len(batch[1]):
                    feed_dict = {self.network.Input_Tensor_Images: batch[0],
                                 self.network.Input_Tensor_Labels: batch[1],
                                 self.network.keep_prob: keep_prob}

                    _, step, summary = sess.run([self.network.train_op_4,
                                                 self.network.global_step,
                                                 self.network.merged], feed_dict)

        print("finished training for {} iters, and {} steps".format(iters,step))


    def optimize(self, batch):
        """ This acuates the optimize funtion and batching function """
        if len(batch[0]) is len(batch[1]):
            Dict = {self.network.Input_Tensor_Images: batch[0],
                    self.network.Input_Tensor_Labels: batch[1],
                    self.network.keep_prob: 0.8}
            self.network.session.run(self.network.optimizer, feed_dict=Dict)
        return batch[2]

    def save_model(self, sess=None):
        """Saves the file using a preset saver func in the builder"""
        model_name = self.dataset.name
        save_path = self.options.save_path
        ext = ".ckpt"
        filename = "{}{}_model_best_acc{}".format(save_path, model_name, ext)
        self.network.saver.save(self.network.session, filename)
        return filename

    def run(self, timeout=5):
        """ This will perform the optimize function"""
        iters = int(1e4)
        epoch = 0
        start_time = time.time()
        start_readout = time.strftime("%a, %d %b %Y %H:%M:%S\n\n",
                                      time.gmtime())
        print("Start Time: {}\nTraining {} Iterations...".format(start_readout,
                                                                 iters))
        for i in tqdm(range(iters)):
            self.werk_done += 1  # tick the clock
            batch = self.dataset.next_batch(self.options.batch_size)
            epoch = self.optimize(batch)
            if i % 25 == 0:
                test_acc, \
                test_loss, \
                train_acc, \
                train_loss = self.feedback(batch)

                if test_loss > self.best_score:
                    self.best_score = test_acc
                    self.last_score = self.werk_done
                    self.save_model()
                if self.werk_done - self.last_score >= timeout:
                    break

        print("Finished Training... Waiting on some Info...")
        batch = self.dataset.next_batch(self.options.batch_size)
        self.feedback(batch)

        end_time = time.time()  # AND STOP THE CLOCK...
        time_dif = end_time - start_time  # do the math
        time_msg = "Time usage: {}\n".format(timedelta(seconds=int(round(time_dif))))  # boom and done.
        print("{}Epochs Complete: {}\nIters Complete: {}".format(time_msg, epoch, self.werk_done))

    def feedback(self, training_batch=False):
        """WORKING THROUGH THE FEEDBACK DEBUGS!! 2_23_17  * finished same day... BOOM..."""

        """This will do a test for acc and loss"""
        testing_start = 110  # should be randowm less than the _num examples
        msg = "Feedback: \n"
        msg += "Total Epochs Complete: {}\n".format(self.dataset._epochs_completed)
        msg += "Total Optimizations Complete: {}\n".format(self.werk_done)
        Testing_set_images = self.dataset.train_images[testing_start:(testing_start + self.options.batch_size)]
        Testing_set_labels = self.dataset.train_labels[testing_start:(testing_start + self.options.batch_size)]
        test_dict = {self.network.Input_Tensor_Images: Testing_set_images,
                     self.network.Input_Tensor_Labels: Testing_set_labels, self.network.keep_prob: 1.0}

        # this is the get_loss function
        test_loss = self.network.loss.eval(feed_dict=test_dict)
        msg += "Test Loss: {:.1f%}\n".format(test_loss)

        # this is the print acc funtion
        test_acc = self.network.session.run(self.network.accuracy, feed_dict=test_dict)
        msg += "Test Acc: {:1%}".format(test_acc)

        if training_batch:
            training_dict = {self.network.Input_Tensor_Images: training_batch[0],
                             self.network.Input_Tensor_Labels: training_batch[1], self.network.keep_prob: 1.0}
            train_loss = self.network.loss.eval(feed_dict=training_dict)
            msg += "Train Loss: {:.1f%}".format(train_loss)
            train_acc = self.network.session.run(self.network.accuracy, feed_dict=training_dict)
            msg += "Train Acc: {:.1f%}".format(train_acc)
            return test_acc, test_loss, train_acc, train_loss
        if self.verbose: print(msg); self.print_weights();
        return test_acc, test_loss

    """NEW STUFF!!! TESTING VERIFICATION! """

    def print_weights(self):
        """This goes through all the Conv Layers and prints their weights"""
        x = 0
        for i in range(len(self.network.conv_layers_wlist)):
            w = self.get_weights(self.network.conv_layers_wlist[x])
            n = self.network.conv_layers_nameslist[x]
            x += 1
            print("\nLayer:{0:s} \n\tWeights:\n\tMean: {1:.5f}, Stdev: {2:.5f}".format(n, w.mean(), w.std()))
        return True

    def get_weights(self, w=None):
        """
        This will return the final layer weights with no params, or that
        layers weights otherwise
        """
        if w is not None:
            x = self.network.session.run(w)
        else:
            x = self.network.session.run(self.network.weights)
        return x

    """ RECLAMATION YARD """

    def BATCH_VERIFY(self, input_tensor, labels, cls_true):
        batch_size = self.options.batch_size
        num_images = len(input_tensor)
        cls_pred = np.zeros(shape=num_images, dtype=np.int)
        i = 0
        while i < num_images:
            j = min(i + batch_size, num_images)  # j is remade frest every loop...
            # feed_dict = self.network.feed_dictionary(test=False,x_batch=input_tensor, y_true_batch=labels)
            feed_dict = {self.network.Input_Tensor_Images: input_tensor[i:j, :],
                         self.network.Input_Tensor_Labels: labels[i:j, :]}
            cls_pred[i:j] = self.network.session.run(self.network.y_pred_cls, feed_dict=feed_dict)
            i = j
        correct = (cls_true == cls_pred)
        return correct, cls_pred

    def run_test(self):
        return self.BATCH_VERIFY(input_tensor=self.dataset.test_images,
                                 labels=self.dataset.test_labels,
                                 cls_true=self.dataset.test_cls)

    # NOT IMPLEMENTED YET...
    def run_valid(self):
        return self.BATCH_VERIFY(input_tensor=self.dataset.valid_images,
                                 labels=self.dataset.valid_labels,
                                 cls_true=self.dataset.valid_cls)

    def run_train(self, ):
        x = self.network.session.run(self.network.accuracy, feed_dict=self.feed_train)  ## TRAINING ACCURACY
        return x

    def challenge(self, ):
        # train_acc   = self.run_train()
        test, _ = self.run_test()
        # valid, _    = self.run_valid()
        test_sum = test.sum()
        # valid_sum   = valid.sum()

        test_acc = float(test_sum) / len(test)
        # valid_acc   = float(valid_sum) / len(valid)
        return train_acc, test_acc  # , valid_acc
