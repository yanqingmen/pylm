'''
connections for rnn
author: hzx
'''
import layer
import node
import opterations
import updator
import util

class EmbdConnect(object):
    '''Embeding connections'''
    def __init__(self, input_size, output_size, input_nodes, output_nodes, alpha=0.01):
        self._input_size = input_size
        self._output_size = output_size
        self._layer = layer.EmbdLayer(input_size, output_size)
        self._updator = updator.EmbdUpdator(alpha)
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes

    def check_data_dimension(self):
        '''check data dimension, for embeding layer,\
         only output data dimension need be checked
        '''
        for output_node in self._output_nodes:
            output_data = output_node.get_data()
            if not output_data.shape[1] == self._output_size:
                raise TypeError("shape of output_data is not fit with output_size: %s:%s",\
                 (output_data.shape[1], self._output_size))

    def forward(self):
        '''forward operation'''
        self._layer.forward(self._input_nodes, self._output_nodes)

    def update(self):
        '''update weights for  embd layer'''
        input_data = self._input_nodes[0]
        gradient_data = self._output_nodes[0]
        self._updator.cal_update_values(input_data, gradient_data)
        self._updator.do_update(self._layer._weights)

    def backprob(self):
        '''backprob operation'''
        self._layer.backprob(self._input_nodes, self._output_nodes)


class RecurrentConnect(object):
    '''connection between recurrent layer'''
    def __init__(self, hiden_size, input_nodes, output_nodes, alpha=0.01, bptt=5):
        self._layer = layer.RecurrentLayer(hiden_size,\
         opterations.np_sigmoid_op, opterations.np_sigmoid_back_op)
        self._updator = updator.WeightsUpdator(input_size, output_size, alpha)
        self._hiden_size = hiden_size
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._bptt = bptt
        shape = self._output_nodes[0].get_data().shape
        self._hist_nodes = [node.Node(util.init_np_zeros_weights(shape[0], shape[1])) for i in xrange(bptt+1)]
        self._gradient_node = util.init_np_zeros_weights(shape[0], shape[1])
        self._gradient_node_t = util.init_np_zeros_weights(shape[0], shape[1])
        self._hist_index = 1

    def forward(self):
        '''forward operation'''
        opterations.np_copy_nodes(self._hist_nodes[0], self._input_nodes[0])
        self._layer.forward(self._input_nodes, self._output_nodes)
        

    def update(self):
        '''update operation'''
        input_data = self._input_nodes[0]
        gradient_data = self._output_nodes[0]
        self._updator.cal_update_values(input_data, gradient_data)
        self._updator.do_update(self._layer._weights)

    def save_history(self, state_node):
        '''save history state'''
        self._hist_nodes.insert(0, self._hist_nodes.pop())    
        t_node = self._hist_nodes[0]
        opterations.np_copy_nodes(state_node, t_node)
        if self._hist_index < self._bptt+1:
            self._hist_index += 1

    def update_history(self, gradient_node):
        '''bptt training via history states'''
        