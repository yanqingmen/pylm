'''evaluation functions'''

import numpy as np


class Entropy(object):
    '''perplexity evaluation for lm model'''
    def __init__(self, batch_size, data_loader, model):
        self._model = model
        self._data_loader = data_loader
        self._batch_size = batch_size
        self._f_index = np.array(range(batch_size))

    def cal_entropy(self):
        '''calculate perplexity of test data'''
        self._data_loader.reset()
        entropy = 0.0
        nums = 0
        for test_data, target_data in self._data_loader:
            predict_data = self._model.predict(test_data)
            predict_probs = predict_data[self._f_index, target_data]
            nums += self._batch_size
            entropy += np.sum(np.log2(predict_probs))

        # print "log probability: ", perplexity
        return -entropy/nums