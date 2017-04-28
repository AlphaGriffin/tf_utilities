import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


# Data Stuffs!!
dataset_paths = "/pub/dataset/mario/"
# input_filename = 'super_set.npz'
input_filename = 'mariokart64_dataset_0.npz'
input_files = os.path.join(dataset_paths, input_filename)
load = np.load(input_files)
imgs = load['images']
labels = load['labels']
dataset_examples = imgs.shape[0]
dataset_h = imgs.shape[1]
dataset_w = imgs.shape[2]
dataset_c = imgs.shape[3]
dataset_classes = labels.shape[1]
dataset_shape = imgs[0].shape
label_example = labels[0]
IMG_W = dataset_w
IMG_H = dataset_h
OUT_SHAPE = dataset_classes
print("Images: {}".format(imgs.shape))
print("Labels: {}".format(labels.shape))
print("Image shape: {}".format(dataset_shape))
print("Label: {}".format(label_example))

# Data Output Stuffs!!
#save_path = '/pub/models/mupen64/mariokart64/'
#save_dirname = 'outputmodel__'
#save_path = os.path.join(save_path, save_dirname)
save_path = '/pub/models/mupen'
save_filename = 'alphagriffin'
print(save_path)


class data_prep(object):
    def __init__(self, images, labels):
        self.index_in_epoch = 0
        self.num_examples = 0
        self.epochs_completed = 0
        self.num_examples = 0
        self.images = images
        self.labels = labels
        self.check_data()

    def check_data(self):
        try:
            assert self.images.shape[0] == self.labels.shape[0]
        except Exception as e:
            print("{} is not {}".format(self.images.shape[0], self.labels.shape[0]))
        self.num_examples = self.images.shape[0]
        return True

    def next_batch(self, batch_size, shuffle=False):
        """ Shuffle is off by default """
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.num_examples)  # should add some sort of seeding for verification
                np.random.shuffle(perm)
                self.images = self.images[perm]
                self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.images[start:end], self.labels[start:end]
batcher = data_prep(imgs, labels)
sess = tf.InteractiveSession()
checkpoint_file = tf.train.latest_checkpoint(save_path)
# filename = '/pub/models/mupen64/mariokart64/outputmodel__/alphagriffin-29901'
# filename2 = '/pub/models/mupen64/mariokart64/outputmodel__/alphagriffin-29901.data-00000-of-00001'
# filename3 = '/pub/models/mupen64/mariokart64/outputmodel__/alphagriffin-29901.index'
print("Checkpoint File: {}".format(checkpoint_file))
meta_path = checkpoint_file + ".meta"
new_saver = tf.train.import_meta_graph(meta_path)
saver = tf.train.import_meta_graph(meta_path)
sess.run(tf.global_variables_initializer())
saver.restore(sess, checkpoint_file)


x = tf.get_collection('x_image')[0]
print(x.name)
y = tf.get_collection('y')[0]
print(y.name)
y_ = tf.get_collection('y_')[0]
print(y.name)
keep_prob = tf.get_collection('keep_prob')[0]
print(keep_prob.name)
loss = tf.get_collection('loss')[0]
print(loss.name)
train = tf.get_collection('train_op')[0]
print(train.name)
learn = tf.get_collection('learn_rate')[0]
print(learn.name)
global_step = tf.get_collection_ref('global_step')[0]
print(global_step.name)

test_batch = batcher.next_batch(42)
loss_value = sess.run(loss, feed_dict={x:test_batch[0], y_: test_batch[1], keep_prob: 1.0})
print("loss: {0:.4f}".format(loss_value))
cur_step = sess.run(global_step)
print("This model has seen {} iterations".format(cur_step))
retrain = True
if retrain:
    iters = 25000
    batch_size = 100
    print("Training For {}".format(iters))
    test = 0
    for i in tqdm(range(iters), desc="Training:"):
        batch = batcher.next_batch(batch_size, True)
        train.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.65})
        # if i % 100 == 0:
        #     step = sess.run(global_step)
        #     loss_value = sess.run(loss, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        # test = epochs = batcher.epochs_completed

        if i % 250 == 0:
            saver.save(sess, save_path + '/alphagriffin', global_step)

for i in range(50, 75):
    test_img = imgs[i]
    feed_dict = {x: [test_img], keep_prob: 1.0}
    joystick = sess.run(y, feed_dict)[0]
    output = [
            int(joystick[0] * 80),
            int(joystick[1] * 80),
            int(round(joystick[2])),
            int(round(joystick[3])),
            int(round(joystick[4])),
        ]
    # plt.imshow(test_img, interpolation='nearest')
    # plt.show()
    print("prediction: {}".format(output))