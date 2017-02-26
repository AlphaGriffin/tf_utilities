#!/usr/bin/env python3
"""
DummyScript.com - 2017
Working Test for Mupen64 Model Training
@#@LICENSE@#@
"""

import os
os.chdir("/home/eric/git/tf_utilities")
import lib.build_network as net               # Plug all the data into Tensorflow... very carefully...
import lib.mupen64 as data                    # Start a custom dataset based on the given dataset + special sauce
import lib.process_network as proc
import lib.options as opts
        
INPUTDATA_IMG = "/home/eric/.local/share/mupen64plus/datasets/mariokart64/_mariokart64_dataset_12_image.npy"
INPUTDATA_LABEL = "/home/eric/.local/share/mupen64plus/datasets/mariokart64/_mariokart64_dataset_12_label.npy"

CONFIG    = opts.options(verbose=False)
DATASET   = data.mupenDataset(INPUTDATA_IMG, INPUTDATA_LABEL, CONFIG)
NETWORK   = net.Build_Adv_Network(DATASET) # THis is where process distribution becomes a reality...Not yet but soon.
WORKER    = proc.Process_Network(NETWORK)
print("##########################################################\n")
print("Size of:")
print("- Training-set:\t\t{}".format(len(DATASET.train_labels)))
print("- Test-set:\t\t{}".format(len(DATASET.test_labels)))
print("##########################################################\n")
WORKER.run(iters=10)
print("##########################################################")
WORKER.end()
print("Thanks! - DummyScript.com")

# check to see if ram is starting to clog up on you...
#os.system("nvidia-smi")