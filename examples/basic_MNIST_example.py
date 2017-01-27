#!/usr/bin/python3
"""
 DUMMYSCRIPT.COM
this is the dummy complete Tensorflow MNIST
"""
import os
os.chdir('/home/eric/git/tf_utilities/') # make sure you are working in the right place
os.system('clear')
""" My Libs """
import lib.mnist as mnist               # Start a custom dataset based on the given dataset + special sauce
import lib.build_network as net         # Plug all the data into Tensorflow... very carefully...
import lib.process_network as proc      # Actually run the TF machine

class options_(object):
    def __init__(self, verbose=True, learning_rate=1e-5, batch_size=100, optimizer=1, entropy=1):
        self.verbose = verbose
        self.model_path = 'models/'
        self.learning_rate = learning_rate
        self.batch_size = batch_size 
        self.optimizer = optimizer
        self.entropy = entropy
        ## CNN options
        self.conv_layers = 2  # of conv layers 2 stock
        self.fc_layers = 2    # of fully connected layers 2 stock     
        self.f_size = 5       # fixed size not file size... silly 5 stock
        self.fc_size = 128    # Max # of elements in FC flatened layers

# HAVE AT THEE!        
TEST_OPTIONS_ONE   = options_(verbose = True, learning_rate=1e-4,batch_size=64,optimizer=2)
options            = TEST_OPTIONS_ONE
MNIST_DATASET      = mnist.MNIST_DATASET(options)
data               = MNIST_DATASET
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