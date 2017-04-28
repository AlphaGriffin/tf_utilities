#!/usr/bin/env python

"""
THIS IS AN EXAMPLE OF THE PROBLEM!!!

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import tensorflow as tf
import numpy as np
from tqdm import tqdm

SavePath = './foo'
File_Name = 'bar'
Meta_Name = File_Name + '.meta'
load = True
if not load:
    # make dataset
    train_X = np.linspace(-1, 1, 101)
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10
    # make metaGraph
    g = tf.Variable(0, trainable=False, name='global_step')
    tf.add_to_collection('g', g)
    X = tf.placeholder("float", name='test_X')
    tf.add_to_collection('X', X)
    Y = tf.placeholder("float", name='test_Y')
    tf.add_to_collection('Y', Y)
    w = tf.Variable(0.0, name="weight")
    b = tf.Variable(0.0, name="bias")
    tf.add_to_collection('w', w)
    tf.add_to_collection('b', b)
    c = tf.square(Y - tf.multiply(X, w) - b)
    tf.add_to_collection('c', c)
    t = tf.train.GradientDescentOptimizer(0.01).minimize(c, global_step=g)
    tf.add_to_collection('t', t)
    # add a save op
    saver = tf.train.Saver() # nope
    # saver = tf.train.Saver([w, b, g]) # still nope
    # saver = tf.train.Saver({"w": w,
    #                        'b': b,
    #                        'g': g},
    #                       max_to_keep=2) # i need more help
    # train a model
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in tqdm(range(300)):
            for (x, y) in zip(train_X, train_Y):
                sess.run(t, feed_dict={X: x, Y: y})
            # save it
            saver.save(sess, os.path.join(SavePath, File_Name), g)
            saver.export_meta_graph(os.path.join(SavePath, Meta_Name), collection_list=['w', 'b', 'g'])
        # print results
        print("Iters Complete: {}".format(sess.run(g)))
        print("Current Bias: {}".format(sess.run(b)))
        print("Current Weight: {}".format(sess.run(w)))
        print("Cost for 2x3: {}".format(sess.run(c, feed_dict={X: 2, Y: 3})))
else:
    sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # get checkpoint state
    checkpoint_file = tf.train.latest_checkpoint(SavePath)
    print("Checkpoint File: {}".format(checkpoint_file))
    # loader function
    meta_path = checkpoint_file + ".meta"
    new_saver = tf.train.import_meta_graph(meta_path, clear_devices=True)
    sess.run(tf.global_variables_initializer())
    new_saver.restore(sess, checkpoint_file)
    all_vars = tf.trainable_variables()
    for v in all_vars:
        print('Trainable: {}: {}'.format(v.name, v.value()))

    # pull variables from graph
    print("\nAll Manually Saved Variables:")
    g = tf.get_collection('g')[0]
    print(g.name)
    w = tf.get_collection('w')[0]
    print(w.name)
    b = tf.get_collection('b')[0]
    print(b.name)
    c = tf.get_collection('c')[0]
    print(c.name)
    t = tf.get_collection('t')[0]
    print(t.name)
    X = tf.get_collection('X')[0]
    print(X.name)
    Y = tf.get_collection('Y')[0]
    print(Y.name)
    # global init

    # print loaded results
    print("\nIters Complete: {}".format(sess.run(g)))
    print("Current Bias: {}".format(sess.run(all_vars[1].value())))
    print("Current Weight: {}".format(sess.run(all_vars[0].value())))
    print("Cost for 2x3: {}".format(sess.run(c, feed_dict={X: 2, Y: 3})))

    # new test
    print()

print("Done || Alphagriffin.com")