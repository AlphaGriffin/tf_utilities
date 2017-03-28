#!/usr/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time
# from datetime import timedelta
# from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import matplotlib
# import matplotlib.pyplot as plt
working_dir = os.path.dirname(os.path.realpath(__file__))
# matplotlib.use('Agg')

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

    def show_img(self, img):
        """trys to represent your img as ascii... lols"""
        GCF = 2
        chars = np.asarray(list(' .,:;irsXA253hMHGS#9B&@'))
        img = np.sum(img, axis=2)
        img -= img.min()
        img = (1.0 - img / img.max()) ** GCF * (chars.size - 1)
        return "\n".join(("".join(r) for r in chars[img.astype(int)]))

    def easy_mode(self, iters=10, keep_prob=0.8):
        start = time.time()
        print("Started New TF Supervised session.\nCurrent time:{}\nProcessing {} Iters".format(start, iters))
        #create a new dir
        self.make_savepath()
        # load the
        # start a graph
        with tf.Graph().as_default():
            print("Started a Graph.")
            # start a session
            sess = tf.Session()
            print("Started a Session")
            # now build the network ...
            self.network.build_default_values()  # ('/gpu:0')
            print("Finished Building the Graph!")
            # run the init script
            sess.run(self.network.init_op)
            print("Initialized the Variables")
            # make a saver... there is a different one in the default values... doesnt matter.
            saver = tf.train.Saver()
            print("Created the Saver Object")
            # make a writer object for tensorboard
            train_writer = tf.summary.FileWriter(self.options.logDir, sess.graph)
            print("Created the Summary Writer")
            # start the training session
            for i in range(iters):
                print("------------")
                print("# Starting Training Iteration: {} of {}".format(i+1, iters))
                # get a batch from the dataset
                batch = self.dataset.next_batch(self.options.batch_size)
                # print("# Get a sample batch:")
                # Sample_img = batch[0][1]
                # Sample_label = batch[1][1]
                # print("# Display Sample Image:")
                # print("Image:\n{}".format(self.show_img(Sample_img)))
                # print("Label:\n{}".format(Sample_label))
                # build the feed dict...
                if len(batch[0]) is len(batch[0]):
                    feed_dict = {self.network.Input_Tensor_Images: batch[0],
                                 self.network.Input_Tensor_Labels: batch[1],
                                 self.network.keep_prob: 0.8}
                    # Don't do this
                    # print("# feed dict:\n{}".format(feed_dict))
                    _, current_step, summary, loss, learn = sess.run([self.network.train_drop_loss,
                                                                      self.network.global_step,
                                                                      self.network.merged,
                                                                      self.network.train_drop_loss,
                                                                      self.network.learn_rate,
                                                                      ], feed_dict)
                    train_writer.add_summary(summary, i)
                    print("\t-Current Step: {}\n\t-Loss: {}\n\t-Learn rate: {}".format(current_step, loss, learn))

            print("finished training")
            # saver.export_meta_graph(os.path.join(self.options.logDir, "filname.meta"))
            # print("saved metagraph")
            tf.train.write_graph(sess.graph_def, self.options.logDir, 'meta')
            print("Wrote Graph to {}/alphagriffin.pbtxt".format(self.options.logDir))
            # save final usable model
            saver.save(sess, os.path.join(self.options.logDir, "Alpha"))
            print("saved final")
            # saver.save(sess, self.options.logDir + "/Alpha", latest_filename="alpha", meta_graph_suffix="griffin",write_meta_graph=True, write_state=True)  # , global_step=self.network.global_step)

        print("finished easy_mode setup test")
        return True

    def run_network(self, iters=10, keep_prob=0.8):
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
        Error_Log = []
        start = time.time()
        # create A BRAND NEW DIRECTORY FOR GODS SAKE!
        self.make_savepath()
        # create supervisor object
        self.bossMan = tf.train.Supervisor(is_chief=True,
                                           logdir=self.options.logDir,
                                           checkpoint_basename='alpha.griffin',
                                           init_op=self.network.init_op,
                                           summary_op=None,  # self.network.merged,
                                           saver=self.network.saver,
                                           global_step=self.network.global_step,
                                           save_model_secs=5)
        try:
            with self.bossMan.managed_session() as sess:
                print("Started New TF Supervised session.\nCurrent time:{}\nProcessing {} Iters".format(start, iters))
                #sumwriter = tf.summary.FileWriter(logdir=self.options.logDir, graph=tf.get_default_graph())
                # tf.train.write_graph(tf.get_default_graph(), self.options.logDir, 'graph.pbtxt')
                try:
                    for i in tqdm(range(iters)):
                        try:
                            #while not self.bossMan.stop():
                            batch = self.dataset.next_batch(self.options.batch_size)
                            # print("iters: {}, epochs: {}".format(i, self.dataset._epochs_completed))
                            if len(batch[0]) is len(batch[1]):

                                feed_dict = {self.network.Input_Tensor_Images: batch[0],
                                             self.network.Input_Tensor_Labels: batch[1],
                                             self.network.keep_prob: 0.8}

                                _, step, summary = sess.run([self.network.train_drop_loss,
                                                             self.network.global_step,
                                                             self.network.merged
                                                             ], feed_dict)
                        except Exception as e:
                            Error_Log.append("Error in looper @ {}: {}, iter: {}\n".format(time.time(), e, i))
                except Exception as e:
                    Error_Log.append("Error with looper @ {}: {}\n".format(time.time(), e))
        except Exception as e:
            Error_Log.append("Error with session @ {}: {}\n".format(time.time(), e))
            # final closing summary write operations
            # draw out 1 in 100 images for tensorboard
            # plot scalars with matplot lib
            # double check savefiles ... maybe should be another job
        msg = "Finished Training new Model for {} Iterations.\nSave Path Dir is: {}\n".format(iters, self.options.logDir)
        if Error_Log:
            msg += "Found Errors in Log: {}".format(len(Error_Log))
            msg += "{}".format("{}".format(e) for e in [Error_Log])
        print(msg)

    def make_savepath(self):
        """
        Verify paths given in the options and make them.
        """
        # print the paths given:
        path = self.options.logDir
        print("Searching for Paths:\nModel Save Path:".format(path))

        if os.path.isdir(path):
            print("found path creating new logging directory.")
            num_logs = len(os.listdir(path))
            new_path = os.path.join(path,"log_{}".format(num_logs))
            os.mkdir(new_path)
            self.options.logDir = new_path
            print("The Save Dir for this session is: {}\nGood Luck!".format(new_path))
            return True
        else:
            print("Something is wrong with your Options.logDir, you entered:\n{}\nPlease use a complete path!".format(
                                                                                                                path))
            try:
                path = "/tmp/TF_MODEL"
                if os.path.isdir("/tmp/TF_MODEL"):
                    num_logs - len(os.listdir(path))
                    new_path = os.path.join(path, "log_{}".format(num_logs))
                    os.mkdir(new_path)
                    self.options.logDir = new_path
                    print("The Save Dir for this session is: {}\nGood Luck!".format(new_path))
                    return True
                else:
                    os.mkdir(path)
                    new_path = os.path.join(path, "Log_0")
                    os.mkdir(new_path)
                    self.options.logDir = new_path
                    print("The Save Dir for this session is: {}\nGood Luck!".format(new_path))
                    return True
            except Exception as e:
                print("SORRY BRO! Nothing is working...")
                return False
                # sys.exit()

    def que_network(self, dir_path):
        # dir_path = './'  # change that to wherever your files are
        ckpt_files = [f for f in os.listdir(dir_path) if os.path.isfile(
            os.path.join(dir_path, f)) and 'ckpt' in f]

        for ckpt_file in ckpt_files:
            saver.restore(sess, dir_path + ckpt_file)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Succesfully loaded model from %s at step=%s.' %
                  (ckpt.model_checkpoint_path, global_step))
        pass

    def load_network(self, path=None):
        """Get most recent version in the log_dir"""
        if path is None:
            path = self.options.logDir
        if os.path.isdir(path):
            print("Searching for most recent Save in:\n{}".format(path))
            num_logs = len(os.listdir(path)) - 1
            # log files start with 0 so... num_logs -1 is used
            folder_name = "log_{}".format(num_logs)
            path = os.path.join(path, folder_name)
        else:
            return False
        # path is now self.options.logDir + log_?
        print("Working Dir: {}".format(path))
        # do the TF train loading thing!