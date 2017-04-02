import tensorflow as tf
import numpy as np

xy = np.loadtxt('input.txt',unpack=True)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]])
y_data = np.array([[0],[1],[1],[0]])

X = tf.placeholder(tf.float32,[None,2])
Y = tf.placeholder(tf.float32,[None,1])

W1 = tf.Variable(tf.random_uniform([2,2],minval=-1.,maxval=1.))
W2 = tf.Variable(tf.random_uniform([2,1],minval=-1.,maxval=1.))

b1=tf.Variable(tf.zeros([2]))
b2=tf.Variable(tf.zeros([1]))

L1 = tf.sigmoid(tf.matmul(X,W1)+b1)
hypo = tf.sigmoid(tf.matmul(L1,W2)+b2)

cost = -tf.reduce_mean(Y*tf.log(hypo)+(1-Y)*tf.log(1-hypo))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    for i in range(10000) :
        sess.run(train,feed_dict={X:x_data,Y:y_data})
        if i%2000==0:
            print(i, sess.run(cost,feed_dict={X:x_data,Y:y_data}))
    predict = tf.equal(tf.floor(hypo+0.5),Y)
    accuracy = tf.reduce_mean(tf.cast(predict,"float"))
    print(sess.run(accuracy,feed_dict={X:x_data,Y:y_data}))
    print(accuracy.eval({X:x_data,Y:y_data}))