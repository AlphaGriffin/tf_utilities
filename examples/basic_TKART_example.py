#!/usr/bin/python3
"""
 DUMMYSCRIPT.COM
this is the dummy complete Tensorflow TKART
"""
import os
os.chdir('/home/eric/git/tf_utilities/') # make sure you are working in the right place
os.system('clear')
""" My Libs """
#import lib.mnist as mnist                # Start a custom dataset based on the given dataset + special sauce
import lib.tkdata as tkdata               # Start a custom dataset based on the given dataset + special sauce
import lib.build_network as net           # Plug all the data into Tensorflow... very carefully...
import lib.process_network as proc        # Actually run the TF machine

class options(object):
    def __init__(self, verbose=True, learning_rate=1e-5, batch_size=42, optimizer=2, entropy=1, conv_layers=2,
                 fc_layers=4, f_size=9, fc_size = 512, iters = 7500,):
        self.verbose = verbose
        self.model_path = 'models/tkart/'
        self.dataset_path = 'dataset/tkart/'
        self.dataset_outpath = 'dataset/tkartbin/'
        self.learning_rate = learning_rate
        self.batch_size = batch_size 
        self.optimizer = optimizer
        self.entropy = entropy
        self.iters = iters
        
        ## CNN options
        self.conv_layers = conv_layers          # of conv layers 2 stock
        self.fc_layers = fc_layers              # of fully connected layers 2 stock     
        self.f_size = f_size                    # fixed size not file size... silly 5 stock
        self.fc_size = fc_size                  # of elements in FC flatened layers
        
        # Image options
        self.image_h = 66
        self.image_w = 200

# HAVE AT THEE!        
TEST_OPTIONS_ONE   = options_(verbose = True, learning_rate=1e-4,batch_size=64,optimizer=2)
options            = TEST_OPTIONS_ONE
data               = tkdata.tkdata_DATASET(options)
print("##########################################################\n")
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train_labels)))
print("- Test-set:\t\t{}".format(len(data.test_labels)))
print("- Validation-set:\t{}".format(len(data.validation_labels)))
print("##########################################################\n")
mnist_network   = net.Build_Adv_Network(options=options,model=data)
Werx            = proc.Process_Network(mnist_network)
print("- Optimizer Settings:\n{}".format(Werx.network.optim()))
print("##########################################################")
acc, werk_done  = Werx.long_haul(iters=1000)
print("Total iters completed: {0}, for an accuracy of {1:.2f} percent".format(werk_done, acc))
print("##########################################################")
Werx.BE_DONE()
print("Actually seriously finished... Alphagriffin.com") 
