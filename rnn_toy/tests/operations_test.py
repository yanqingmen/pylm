'''
test operation functions
author: hzx
'''
import sys
import unittest
sys.path.append('../')
import numpy as np
from numpy import random
import operations
import node

class TestOperations(unittest.TestCase):
    def test_emb_op(self):
        weights = random.random((10, 10))
        source = np.array([0, 1, 3, 5, 7])
        target = np.zeros((5, 10))
        operations.np_emb_op(weights, node.Node(source), node.Node(target))
        