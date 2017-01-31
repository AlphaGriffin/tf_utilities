 #!/usr/bin/python2
from utils import Data
import model
import tensorflow as tf

# Load Training Data
data = Data()   # is another script here

# Start session
sess = tf.InteractiveSession()

# Learning Functions
L2NormConst = 0.001 # ??
train_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.square(tf.sub(model.y_, model.y))) +\
			tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) *\
			L2NormConst


train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

sess.run(tf.global_variables_initializer())

# Training loop variables
epochs = 99
batch_size = 42
num_samples = data.num_examples
step_size = int(num_samples / batch_size) # this was hardcoded in other examples

for epoch in range(epochs):
    for i in range(step_size):
        batch = data.next_batch(100)

        train_step.run(feed_dict={model.x: batch[0], model.y_: batch[1], model.keep_prob: 0.8}) #! Hardcoded value

        if i%25 == 0:
          loss_value = loss.eval(feed_dict={model.x:batch[0], model.y_: batch[1], model.keep_prob: 1.0})  #! hardcoded value
          print("epoch: {} step: {} loss: {}".format(epoch, epoch * batch_size + i, loss_value))

# Save the Model
saver = tf.train.Saver()
saver.save(sess, "model.ckpt")
