'''
utils for rnn
author: hzx
'''
from numpy import random


def init_np_weights(rows, cols):
    '''initialize numpy tensor weights'''
    return random.randn(rows, cols)
