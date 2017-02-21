import tensorflow as tf
import time
from datetime import timedelta
#from sklearn.metrics import confusion_matrix
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
#import numpy as np

class Process_Network(object):
    """ This class handles Opimization / Visualization for TF Models """
    def __init__(self, network, ipy=None):
        self.werk_done = 0
        self.ipy = ipy   
        self.network = network
        
        # get dataset from network
        self.dataset = network.dataset
        # get options from network
        self.options = network.options
        # get verbose
        self.verbose = self.options.verbose
        
        if self.verbose:
            from pprint import pprint
            pprint (vars(self.dataset))
            pprint (vars(self.network))
        # setup a better output
        self.progress = tqdm
        
    
    def BE_DONE(self):
        """ This is run at the end of a TF script session"""
        self.network.session.close() 
        
    def run(self, epochs=1):
        """ This will perform the optimize function"""
        # always run a .. .run timer... its in the name
        start_time = time.time() 
        current_epoch = self.dataset._epochs_completed
        goal_train = current_epoch + epochs
        print("{}\nTraining {} epochs...".format(start_time,epochs))
        while current_epoch <= goal_train:
            batch = self.dataset.next_batch(self.options.batch_size)
            current_epoch = batch[2]
            if len(batch[0]) is len(batch[1]):
                # setup up a dict
                Dict = {self.network.Input_Tensor_Images: batch[0], self.network.Input_Tensor_Labels: batch[1], self.network.keep_prob: 0.7}
                
                # This is the optimize function
                self.network.session.run(self.network.optimizer, feed_dict=Dict)
                self.werk_done += 1
        print("Finished Training... Waiting on some Info...")
        batch = self.dataset.next_batch(self.options.batch_size)
        #self.feedback(batch)
        end_time = time.time()             # AND STOP THE CLOCK...
        time_dif = end_time - start_time   # do the math
        time_msg = "Time usage: " + str(timedelta(seconds=int(round(time_dif))))   # boom and done.
        print(time_msg, ", Iters Complete: %s" % self.werk_done)
   
    def feedback(self,training_batch=False):
        """This will do a test for acc and loss"""
        testing_start = 110 # should be randowm less than the _num examples
        msg = "Feedback: \n"
        msg += "Total Epochs Complete: {}\n".format(self.dataset._epochs_completed)
        msg += "Total Optimizations Complete: {}\n".format(self.werk_done)
        
        Testing_set_images = self.dataset.train_images[testing_start:(testing_start+self.options.batch_size)]
        Testing_set_labels = self.dataset.train_labels[testing_start:(testing_start+self.options.batch_size)]
        
        test_dict = {self.network.Input_Tensor_Images: Testing_set_images, self.Input_Tensor_Labels: Testing_set_labels, self.keep_prob: 1.0}
        
        # this is the get_loss function
        loss = self.network.loss.eval(feed_dict=test_dict)
        msg += "Test Loss: {:.5}\n".format(loss)
        
        # this is the print acc funtion
        test_acc = self.network.session.run(self.network.accuracy, feed_dict=test_dict)
        msg += "Test Acc: {:.3}\n".format(test_acc)
        
        # there should be a get confusted 
        # and a print weights
        if training_batch:
            training_dict = {self.network.Input_Tensor_Images: training_batch[0], self.Input_Tensor_Labels: training_batch[1], self.keep_prob: 1.0}
            loss = self.network.loss.eval(feed_dict=training_dict)
            msg += "Train Loss: {:.5}\n".format(loss)
            test_acc = self.network.session.run(self.network.accuracy, feed_dict=training_dict)
            msg += "Train Acc: {:.3}\n".format(loss)
            print(msg)
        

    # RETURNING w for IPY.plot_weights 
    def get_weights(self):
        """This should still work"""
        x = self.network.session.run(self.network.weights)
        return x
        
    # RETURNING ConfusionMatrix and NumClasses for IPY.plot_confused 
    def get_confused(self): pass
    """
    def get_confused(self):
        #get num_classes
        num_classes = self.model.num_classes
        # Get the true classifications for the test-set.
        cls_true = self.model.test_cls
        # Get the predicted classifications for the test-set.
        cls_pred = self.network.session.run(self.network.y_pred_cls, feed_dict=self.network.feed_test)
        cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
        if self.options.verbose: print(cm);
        return cm, num_classes
    """
    # RETURNING IMAGES, CLS_TRUE, CLS_PRED for IPY.SIMPLE_PLOT
    def get_example_errors(self):
        """This should still work"""
        self.print_acc()
        correct, cls_pred = self.network.session.run([self.network.correct_prediction, self.network.y_pred_cls],
                                                     feed_dict=self.network.feed_test)
        incorrect = (correct == False)
        images = self.dataset.images_test[incorrect]
        cls_pred = cls_pred[incorrect]
        cls_true = self.dataset.cls_true[incorrect]
        n = min(9, len(images))       
        return images[0:n], cls_true[0:n], cls_pred[0:n]
    
    # This is A Verbose Output of All Available Sources
    def show_Verbose(self):
       ## GET SOME DATAS
       cm, num_classes = self.get_confused()
       img, cls_t,cls_p = self.get_example_errors()
       ## Bring in IPY
       self.ipy.plot_confused(cm,num_classes)
       self.ipy.simple_plot(images=img, cls_true=cls_t, cls_pred=cls_p)
    
    def print_samples(self):
        """This should tell you if you have congruent img:labels"""
        x = self.dataset.train_images[0:9]
        y = self.dataset.train_labels[0:9]
        index = 0
        for i in x:
            print("img {} label = {}".format(i,y[index]))
            index += 1