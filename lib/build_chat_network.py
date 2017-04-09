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

import os, sys, datetime, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import collections
import random
import tensorflow as tf
import numpy as np
import ag.logging as log
log.set(4)

def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


class App(object):
    def __init__(self):
        self.n_input = 3
        self.n_hidden = 512
        self.n_classes = 10  # MNIST total classes (0-9 digits)
        self.logs_path = '/home/eric/repos/pycharm_repos/TF_Curses/ag/tf_curses/chatbot/models'
        self.train_iters = int(1e5)

    def main(self, args):
        log.info("TESTRUN -")
        # get a text file... say sample.txt
        try:
            file = args[1]
        except:
            file = None
        if file is None:
            file = "../text/sample.txt"

        # Get some data
        log.info("Opening File: {}".format(file))
        sample_set = self.get_text_file(file)
        log.debug("\{}".format(sample_set.text))

        # clean your data
        log.info("building dictionary")
        sample_set = self.build_dataset(sample_set)
        log.debug("Dict len = {}".format(sample_set.dict_len))

        # build a tensorboard
        log.info("build tensorflow network")
        network = self.build_network(sample_set)
        log.debug("Working with Final Layer {}".format(network.final_layer))

        # do some work
        msg = "Train Iters: {}".format(self.train_iters)
        log.info("Training Details:\n{}".format(msg))
        final_loss, average_acc = self.process_network(sample_set, network)
        log.info("Finished Training! Final Loss: {} Accuracy: {}".format(final_loss, average_acc))
        return True

    def get_text_file(self, file_):
        if not os.path.isfile(file_):
            log.warn("{} is an invalid path".format(file_))
            return False
        class sample_text(): pass

        sample_text.text = open(file_, "r").read()
        sample_text.len = len(sample_text.text)
        sample_text.chars = sorted(list(set(sample_text.text)))
        sample_text.len_chars = len(sample_text.chars)
        sample_text.textlines = open(file_, "r").readlines()
        sample_text.nwords = 0
        sample_text.word_set = []
        for l in sample_text.textlines:
            w = l.split()
            for word in w:
                sample_text.word_set.append(word.lower())
                sample_text.nwords += 1
        # print("Sample Text: \n\t{}".format(sample_text))
        return sample_text

    def build_dataset(self, sample_set):
        sample_set.count = collections.Counter(sample_set.word_set).most_common()
        sample_set.dictionary = dict()
        log.debug("adding word at pos. word[pos]")
        for word, _ in sample_set.count:
            cur_len = len(sample_set.dictionary)
            log.debug("{} [{}]".format(word, cur_len))
            sample_set.dictionary[word] = cur_len
            sample_set.reverse_dictionary = dict(zip(sample_set.dictionary.values(),
                                                     sample_set.dictionary.keys()))
        sample_set.dict_len = len(sample_set.dictionary)
        return sample_set

    def RNN(self, x, weights, biases):

        # reshape to [1, n_input]
        x = tf.reshape(x, [-1, self.n_input])

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split(x, self.n_input, 1)

        # 1-layer LSTM with n_hidden units.
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        #rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(self.n_hidden), \
        #           tf.contrib.rnn.BasicLSTMCell(self.n_hidden)])

        # generate prediction
        outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    def build_network(self, sample_set):
        class training_ops(): pass
        # RNN output node weights and biases
        # tf Graph input
        with tf.variable_scope("inputs") as scope:
            training_ops.global_step = tf.Variable(0, trainable=False, name='global_step')
            training_ops.learn_rate = tf.train.exponential_decay( 0.1,
                                                                  training_ops.global_step,
                                                                  .000005,
                                                                  0.87,
                                                                  staircase=True,
                                                                  name="Learn_decay"
                                                                  )
            tf.add_to_collection("global_step", training_ops.global_step)
            tf.add_to_collection("learn_rate", training_ops.learn_rate)
            tf.summary.scalar("global_step", training_ops.global_step)
            tf.summary.scalar("decay_rate", training_ops.learn_rate)
            tf.summary.histogram('decay_rate', training_ops.learn_rate)

            training_ops.input_word = tf.placeholder("float", [None, self.n_input, 1])
            training_ops.input_label = tf.placeholder("float", [None, sample_set.dict_len])
            tf.add_to_collection("input_word", training_ops.input_word)
            tf.add_to_collection("input_label", training_ops.input_label)

        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, sample_set.dict_len]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([sample_set.dict_len]))
        }

        # learning_rate = 0.001

        training_ops.final_layer = self.RNN(training_ops.input_word, weights, biases)
        tf.add_to_collection("final_layer", training_ops.final_layer)
        # Evaluate model
        training_ops.correct_pred = tf.equal(tf.argmax(training_ops.final_layer, 1), tf.argmax(training_ops.input_label, 1))
        training_ops.accuracy = tf.reduce_mean(tf.cast(training_ops.correct_pred, tf.float32))
        tf.summary.scalar("accuracy", training_ops.accuracy)
        tf.summary.histogram('accuracy', training_ops.accuracy)
        tf.add_to_collection("correct_pred", training_ops.correct_pred)
        tf.add_to_collection("accuracy", training_ops.accuracy)

        # Loss and optimizer
        training_ops.cost = tf.reduce_mean( \
                            tf.nn.softmax_cross_entropy_with_logits(logits=training_ops.final_layer,
                                                                    labels=training_ops.input_label))
        tf.summary.scalar("cost", training_ops.cost)
        tf.summary.histogram('cost', training_ops.cost)
        tf.add_to_collection("cost", training_ops.cost)
        training_ops.optimizer = tf.train.RMSPropOptimizer(learning_rate=training_ops.learn_rate) \
                                                            .minimize(training_ops.cost)

        tf.add_to_collection("optimizer", training_ops.optimizer)
        training_ops.init_op = tf.global_variables_initializer()
        training_ops.saver = tf.train.Saver()
        training_ops.merged = tf.summary.merge_all()
        # DO MERGED SUMMARY!!
        return training_ops

    def process_network(self, sample_set, network):

        # DEFINES!!
        training_data = sample_set.word_set
        dictionary = sample_set.dictionary
        reverse_dictionary = sample_set.reverse_dictionary
        n_input = self.n_input
        vocab_size = sample_set.dict_len

        # start here
        start_time = time.time()
        session = tf.Session()
        session.run(network.init_op)
        writer = tf.summary.FileWriter(self.logs_path)
        step = 0
        offset = random.randint(0, n_input + 1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0
        display_step = 50
        msg = "step: {}, offset: {}, acc_total: {}, loss_total: {}".format(step,offset, acc_total, loss_total)
        log.debug("Starting the Train Session:\n{}".format(msg))
        # start by adding the whole graph to the Tboard
        writer.add_graph(session.graph)

        while step < self.train_iters:
            # Generate a minibatch. Add some randomness on selection process.
            if offset > (len(training_data) - end_offset):
                offset = random.randint(0, n_input + 1)

            symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
            symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

            symbols_out_onehot = np.zeros([vocab_size], dtype=float)
            symbols_out_onehot[dictionary[str(training_data[offset + n_input])]] = 1.0
            symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

            feed_dict = {network.input_word: symbols_in_keys,
                         network.input_label: symbols_out_onehot}

            _, acc, loss, onehot_pred, _step, summary = session.run([network.optimizer,
                                                                    network.accuracy,
                                                                    network.cost,
                                                                    network.final_layer,
                                                                    network.global_step,
                                                                    network.merged
                                                                     ],
                                                                    feed_dict=feed_dict)

            network.global_step += 1
            loss_total = loss
            acc_total += acc
            if (step + 1) % display_step == 0:
                # acc pool
                acc_total = (acc_total * 100) / display_step
                writer.add_summary(summary, global_step=_step)
                symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                symbols_out = training_data[offset + n_input]
                symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval(session=session))]
                msg = ' "{}" *minus* "{}" *equals* "{}"\n'.format(symbols_in, symbols_out, symbols_out_pred)
                msg += "step: {0:}, offset: {1:}, acc_total: {2:.2f}, loss_total: {3:.2f}\n".format(_step,
                                                                                             offset,
                                                                                             acc_total,
                                                                                             loss_total)
                log.info("Checking the Train Session:\n{}".format(msg))
                network.saver.save(session, self.logs_path, global_step=_step)
                acc_total = 0
            step += 1
            offset += (n_input + 1)
        log.info("Optimization Finished!")
        log.debug("Elapsed time: {}".format(elapsed(time.time() - start_time)))
        return(loss_total, acc_total)
        session.close()


if __name__ == '__main__':
    try:
        os.system('clear')
        app = App()
        if app.main(sys.argv):
            sys.exit("Thanks A lot for trying Alphagriffin.com")
        log.warn("Alldone! Alphagriffin.com")

    except KeyboardInterrupt:
        os.system('clear')
        sys.exit("AlphaGriffin.com")