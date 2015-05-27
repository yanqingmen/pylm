'''evaluation functions'''

import numpy as np


class Perplexity(object):
    '''perplexity evaluation for lm model'''
    def __init__(self, batch_size, test_batchs, target_batchs, model):
        self._model = model
        self._test_batchs = test_batchs
        self._target_batchs = target_batchs
        self._batch_size = batch_size
        self._f_index = np.array(range(batch_size))

    def cal_perplexity(self):
        '''calculate perplexity of test data'''
        perplexity = 0.0
        nums = 0
        for test_data, target_data in zip(self._test_batchs, self._target_batchs):
            predict_data = self._model.predict(test_data)
            predict_probs = predict_data[self._f_index, target_data]
            nums += self._batch_size
            perplexity += np.sum(np.log2(predict_probs))

        return np.exp2(-perplexity/nums)