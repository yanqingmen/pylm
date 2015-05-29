'''
weights or bias updators for rnn
author: hzx
'''
import numpy as np
import util


class BiasUpdator(object):
    '''bias updator, handle the tmp bias data for certain layer'''
    def __init__(self, bias_size, alpha):
        self._bias_t = util.init_np_zeros_weights(1, bias_size)
        self._alpha = alpha

    def cal_update_values(self, gradient):
        '''calculate update values for bias via gradient'''
        self._bias_t[:] = np.sum(np.sum(gradient, axis=1))

    def do_update(self, bias):
        '''do update for given bias'''
        bias += self._bias_t * self._alpha


class WeightsUpdator(object):
    '''normal weights updator, handle the tmp weights data for certain layer'''
    def __init__(self, input_size, output_size, alpha):
        self._weights_t = util.init_np_zeros_weights(input_size, output_size)
        self._alpha = alpha
        self._beta = 0.0000001

    def cal_update_values(self, input_data, gradient_data):
        '''calculate update values for bias via gradient'''
        np.dot(input_data.T, gradient_data, self._weights_t)

    def do_update(self, weights):
        '''do update for given weights'''
        self._weights_t *= self._alpha
        weights *= (1.0 - self._beta *self._alpha)
        weights += self._weights_t

class EmbdUpdator(object):
    '''updator for embding layer weights'''
    def __init__(self, alpha):
        self._alpha = alpha
        self._input_data = None
        self._gradient_data = None

    def cal_update_values(self, input_data, gradient_data):
        '''just save input_data and gradient_data'''
        self._input_data = input_data
        self._gradient_data = gradient_data

    def do_update(self, weights):
        '''do update for given weights'''
        weights[self._input_data] += self._gradient_data * self._alpha
