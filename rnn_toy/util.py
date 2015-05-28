'''
utils for rnn
author: hzx
'''
from numpy import random
import numpy as np
import node

def init_np_zeros_weights(rows, cols):
    '''initialize numpy zero tensor'''
    return np.zeros((rows, cols))

def init_np_weights(rows, cols):
    '''initialize numpy tensor weights'''
    return random.randn(rows, cols)

def flat_target_data(target_data, flatted_target_data):
    '''flat target data'''
    f_index = np.array(range(target_data.shape[0]))
    flatted_target_data[f_index, target_data] = 1

def np_clip(sdata):
    '''make abs value of sdata no more than 50, for numerical stability'''
    np.clip(sdata, -50, 50, sdata)


def np_sigmoid(sdata):
    '''cal sigmoid for numpy data'''
    np_clip(sdata)
    np.negative(sdata, sdata)
    np.exp(sdata, sdata)
    sdata += 1
    np.reciprocal(sdata, sdata)


def np_sigmoid_backprob(grad_data, state_data, output_data):
    '''
    cal backprob of sigmoid func for numpy data
    output_data = grad_data * state_data * (1 - state_data)
    * this function will change the value of both sdata and tdata
    '''
    np.multiply(grad_data, state_data, output_data)
    output_data *= (1 - state_data)


def np_softmax(sdata):
    '''cal softmax for numpy data'''
    np_clip(sdata)
    np.exp(sdata, sdata)
    row_sum = np.sum(sdata, axis=1)
    for i in xrange(sdata.shape[0]):
        sdata[i] /= row_sum[i]

def np_cal_gradient(output_data, target_data, gradient_data):
    '''gradient_data = output_data - target_data'''
    np.add(-output_data, target_data, gradient_data)

def create_np_node(batch_size, data_size, init_random=False, grad=None):
    '''create numpy data node'''
    if init_random:
        data = init_np_weights(batch_size, data_size)
    else:
        data = init_np_zeros_weights(batch_size, data_size)
    if grad is not None:
        grad_data = grad
    else:
        grad_data = init_np_zeros_weights(batch_size, data_size)
    return node.Node(data, grad=grad_data)
    