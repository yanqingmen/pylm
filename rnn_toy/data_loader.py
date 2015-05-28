'''data loader for big data base'''
#-*- coding: utf-8 -*-

import numpy as np

class LmDataLoader(object):
    '''data loader for training or testing language model'''
    def __init__(self, corpus_file, batch_size):
        self._corpus_file = open(corpus_file, "r")
        self._tmp_data = None
        self._tmp_index = 0
        self._batch_size = batch_size
        self._previous = 0

    def __iter__(self):
        return self

    def reset(self):
        '''reset data loader'''
        self._corpus_file.seek(0)

    def read(self):
        '''read next id'''
        if self._tmp_data is None or self._tmp_index > len(self._tmp_data):
            self._tmp_index = 0
            self._tmp_data = None
            try:
                line = self._corpus_file.next()
                self._tmp_data = map(int, line.strip().split())
            except StopIteration:
                return None
        if self._tmp_index == len(self._tmp_data):
            self._tmp_index += 1
            data = 0
        else:
            data = self._tmp_data[self._tmp_index]
            self._tmp_index += 1
        prev = self._previous
        self._previous = data
        return (prev, data)

    def next(self):
        '''get next batch'''
        input_batch = []
        target_batch = []
        while True:
            data = self.read()
            # print data, batch
            if data is None:
                raise StopIteration
            # if data is None and len(input_batch) == 0:
            #     raise StopIteration
            # if data == None:
            #     data = (0, 0)
            input_batch.append(data[0])
            target_batch.append(data[1])
            if len(input_batch) == self._batch_size:
                break

        return (np.array(input_batch), np.array(target_batch))
