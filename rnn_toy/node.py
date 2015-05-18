'''
rnn_toy a toy implementation for rnn
nodes for data handling (input data, output data, activation, gradient, etc. )
author: hzx
'''
#-*- coding: utf-8 -*-

class Node(object):
    '''node class'''
    def __init__(self, data, grad=None, info=""):
        self._data = data
        self._grad = grad
        self._info = info

    def get_data(self):
        '''get data'''
        return self._data

    def get_grad(self):
        '''get gradient'''
        return self._grad

    def get_info(self):
        '''get node info'''
        return self._info
        