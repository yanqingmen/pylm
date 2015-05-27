'''
a toy rnn implmentation
'''
#-*- coding:utf-8 -*-

import connections
import node
import util

class SimpleRnn(object):
    '''simple rnn'''
    def __init__(self, batch_size, emb_size, hiden_size, output_size, alpha=0.01, bptt=10):
        emb_connect = connections.EmbdConnect(emb_size, hiden_size, batch_size, alpha)
        rec_connect = connections.RecurrentConnect(hiden_size, batch_size, alpha, bptt)
        full_connect = connections.FullCLayerConnect(hiden_size, output_size, batch_size, alpha)
        softmax_connect = connections.SoftMaxConnect()
        self.connects = [emb_connect, rec_connect, full_connect, softmax_connect]
        self.flatted_target_data = util.init_np_zeros_weights(batch_size, output_size)
        self.connects[-1].set_target(self.flatted_target_data)
        #link
        for i in xrange(1, len(self.connects)):
            self.connects[i].link_to_previous(self.connects[i-1])

    def predict(self, input_data):
        '''predict via input_data'''
        input_nodes = [node.Node(input_data)]
        self.connects[0].set_input_nodes(input_nodes)
        for connect in self.connects:
            connect.forward()

        output_data = self.connects[-1].get_output_nodes()[0].get_data()
        return output_data

    def backprob(self, target_data):
        '''backprob after predict'''
        self.flatted_target_data[:] = 0
        util.flat_target_data(target_data, self.flatted_target_data)
        for i in xrange(len(self.connects)-1, -1, -1):
            self.connects[i].backprob()

    def train_one_batch(self, input_data, target_data):
        '''train one batch'''
        self.predict(input_data)
        self.backprob(target_data)

    def train_one_round(self, input_batchs, target_batchs):
        '''train one round'''
        assert len(input_batchs) == len(target_batchs)

        for input_data, target_data in  zip(input_batchs, target_batchs):
            self.train_one_batch(input_data, target_data)
        