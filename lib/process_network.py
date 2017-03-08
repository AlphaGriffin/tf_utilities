import tensorflow as tf
import time
from datetime import timedelta
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

working_dir = os.path.dirname(os.path.realpath(__file__))
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

    def run_network(self, iters=5000, keep_prob=0.8):
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
                                           summary_op=self.network.merged,
                                           saver=self.network.saver,
                                           global_step=self.network.global_step,
                                           save_model_secs=60)
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

    def que_network(self): pass

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

