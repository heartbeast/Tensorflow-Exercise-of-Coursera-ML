
from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import datetime
import sys,os
sys.path.append(os.path.join(sys.path[0], "../"))
from util import featureStandardize

data=numpy.loadtxt(sys.path[0]+"/ex1data2.txt", delimiter=',')
print(type(data),data.shape,data[0],data[1])
train_X = data[:,0:2]
train_Y = data[:,2]
print(type(train_X),type(train_Y), train_X.shape, train_Y.shape)
print("init train_X:", train_X[0:5])
print("init train_Y:", train_Y[0:5])

# featureNormalize
train_X = featureStandardize(train_X)
print("after norm:\n", train_X[0:5])

m_samples = train_X.shape[0]
n_feature = train_X.shape[1]
print("m_samples:%s, n_feature:%s" %(m_samples,n_feature))
train_Y = train_Y.reshape(m_samples,1)   # train_Y is m*1


ones    = numpy.ones([m_samples,1])
train_X = numpy.c_[ones,train_X]   # add 1 column
print("train_X.shape:", train_X.shape)
print("after add 1:\n", train_X[0:5])

# tf Graph Input
X = tf.placeholder(tf.float32, [m_samples, n_feature+1])
Y = tf.placeholder(tf.float32, [m_samples,1])

# Set model weights
theta = tf.Variable(tf.zeros([n_feature+1, 1]), name="theta")
print("theta:", theta)


# Construct a linear model
pred = tf.matmul(X, theta)
# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*m_samples)
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
    t = sess.run(theta)
    print("init value, cost={:.1f}".format(c), "theta=%.2f,%.2f,%.2f"%tuple(t))

    # start = datetime.datetime.now()
    # Fit all training data
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            trained_cost = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            trained_theta = sess.run(theta)
            print("Epoch:%04d" % (epoch+1), "cost=%.1f"%trained_cost, " theta=%.2f,%.2f,%.2f"%tuple(trained_theta))

    # end = datetime.datetime.now()
    # print("duration:",end-start)

    print("Optimization Finished!")
    trained_cost  = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    trained_theta = sess.run(theta)
    print("Trained cost={:.1f}".format(trained_cost), "theta=%f,%f,%f"%tuple(trained_theta))
