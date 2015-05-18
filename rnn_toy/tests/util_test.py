'''
test util functions
author: hzx
'''
import sys
import unittest
sys.path.append('../')
import util
import numpy as np
from numpy import random

class TestUtilMethods(unittest.TestCase):
    def test_sigmoid(self):
        source = random.random((5, 50))
        reference = 1/(1+np.exp(-source))
        util.np_sigmoid(source)
        self.assertTrue(np.max(reference - source) < 1e-4)

    def test_sigmoid_backprob(self):
        source = np.zeros((10, 10))
        source[:] = 1.0
        target = np.zeros((10, 10))
        target[:] = 0.5

        reference = target * source * (1 - source)
        util.np_sigmoid_backprob(source, target)
        self.assertTrue(np.max(reference - target) < 1e-4)

    def test_softmax(self):
        source = random.random((10, 5))
        reference = np.exp(source)
        for i in xrange(reference.shape[0]):
            reference[i] /= np.sum(reference[i])
        util.np_softmax(source)
        self.assertTrue(np.max(reference - source) < 1e-4)


if __name__ == '__main__':
    unittest.main()