# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def plot_xy(X,Y,m):
    for i in range(m):
        if Y[i]==1:
            plt.plot(X[i,0], X[i,1], 'r+', linewidth=2.5)
        else:
            plt.plot(X[i,0], X[i,1], 'b.', linewidth=0.5)
    # plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    # plt.legend()
    plt.show()

# feature Scaling
def featureStandardize(arr):
    div = len(arr.shape)
    if div==1: # vector
        axis=None
    elif div==2: # matrix
        axis=0
    else:
        raise ValueError("only support vector and matrix input")
    mean = np.mean(arr,axis)
    std =  np.std(arr,axis, ddof=1)
    return (arr-mean) / std

def featureNormalize(arr):
    div = len(arr.shape)
    if div==1: # vector
        axis=None
    elif div==2: # matrix
        axis=0
    else:
        raise ValueError("only support vector and matrix input")
    mean = np.mean(arr,axis)
    npmax =  np.ma.max(arr,axis)
    npmin =  np.ma.min(arr,axis)
    return (arr-mean) / (npmax-npmin)

# return a dictionary
def loadmat(fname):
    import scipy.io
    data=scipy.io.loadmat(fname)
    return data

if __name__ == '__main__':
    import numpy
    import matplotlib.pyplot as plt  

    train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    # plt.figure(1)  
    plt.plot(train_X,train_Y)
    plt.show()  

    # sio.savemat('saveddata.mat', {'xi': xi,'yi': yi,'ui': ui,'vi': vi})  
