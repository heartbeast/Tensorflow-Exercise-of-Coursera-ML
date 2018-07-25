from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

import sys

data=numpy.loadtxt(sys.path[0]+"/ex1data1.txt", delimiter=',')
print(type(data),data.shape,data[0],data[1])
train_X = data[:,0]
train_Y = data[:,1]
print(type(train_X),type(train_Y), train_X.shape, train_Y.shape)
print("init train_X:", train_X[0:5])
print("init train_Y:", train_Y[0:5])

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    print("\n======================= tf session start =======================\n")
    
    training_epochs = 3000
    display_step = 100

    # Run the initializer
    sess.run(init)
    c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
    print("init value, cost={:.6f}".format(c))

    # Fit all training data
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
