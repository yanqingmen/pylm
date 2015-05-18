'''
utils for rnn
author: hzx
'''
from numpy import random
import numpy as np

def init_np_zeros_weights(rows, cols):
    '''initialize numpy zero tensor'''
    return np.zeros((rows, cols))

def init_np_weights(rows, cols):
    '''initialize numpy tensor weights'''
    return random.randn(rows, cols)


def np_sigmoid(sdata):
    '''cal sigmoid for numpy data'''
    np.negative(sdata, sdata)
    np.exp(sdata, sdata)
    sdata += 1
    np.reciprocal(sdata, sdata)


def np_sigmoid_backprob(sdata, tdata):
    '''
    cal backprob of sigmoid func for numpy data
    backprob = tdata * [sdata * (1 - sdata)]
    * this function will change the value of both sdata and tdata
    '''
    tdata *= sdata
    tdata *= (1 - sdata)


def np_softmax(sdata):
    '''cal softmax for numpy data'''
    np.exp(sdata, sdata)
    row_sum = np.sum(sdata, axis=1)
    for i in xrange(sdata.shape[0]):
        sdata[i] /= row_sum[i]
