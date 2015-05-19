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
    def __init__(self, input_size, output_size, batch_size, alpha=0.01):
        self._input_size = input_size
        self._output_size = output_size
        self._batch_size = batch_size
        self._layer = layer.EmbdLayer(input_size, output_size)
        self._updator = updator.EmbdUpdator(alpha)
        #input nodes will be set by link to previous connection or by outside data source
        self._input_nodes = None
        #create output data nodes
        self._output_nodes = [util.create_np_node(batch_size, output_size)]

    def check_data_dimension(self):
        '''check data dimension, for embeding layer,\
         only input data dimension need be checked
        '''
        input_node = self._input_nodes[0]
        if not input_node.get_data().shape[0] == self._batch_size:
            raise TypeError(" shape of input_data is not fit with batch_size, %s:%s",\
             (input_node.get_data().shape[0], self._batch_size))

    def set_input_nodes(self, input_nodes):
        '''set input nodes for this connection'''
        self._input_nodes = input_nodes

    def link_to_previous(self, prev_conn):
        '''link this connection to previous connection'''
        self._input_nodes = prev_conn.get_output_nodes()

    def get_output_nodes(self):
        '''get output nodes of this connection'''
        return self._output_nodes

    def forward(self):
        '''forward operation'''
        self._layer.forward(self._input_nodes, self._output_nodes)

    def update(self):
        '''update weights for  embd layer'''
        input_data = self._input_nodes[0].get_data()
        gradient_data = self._output_nodes[0].get_grad()
        self._updator.cal_update_values(input_data, gradient_data)
        self._updator.do_update(self._layer._weights)

    def backprob(self):
        '''backprob operation, nothing'''
        pass


class RecurrentConnect(object):
    '''connection between recurrent layer'''
    def __init__(self, hiden_size, batch_size, alpha=0.01, bptt=5):
        self._layer = layer.RecurrentLayer(hiden_size,\
         opterations.np_sigmoid_op, opterations.np_sigmoid_back_op)
        self._updator = updator.WeightsUpdator(hiden_size, hiden_size, alpha)
        self._hiden_size = hiden_size
        # recurrent layer has a defualt data node which shared by input and output
        self._input_nodes = [util.create_np_node(batch_size, hiden_size)]
        # besides the default data node, output nodes also include a tmp data node for save tmp data
        self._output_nodes = [self._input_nodes[0], util.create_np_node(batch_size, hiden_size)]
        self._bptt = bptt

        self._hist_nodes = self._create_hist_nodes(batch_size, hiden_size, gradient_data, bptt)
        self._hist_index = 1

    def forward(self):
        '''forward operation'''
        self._layer.forward(self._input_nodes, self._output_nodes)
        self.save_history(self._output_nodes[0])

    def update(self):
        '''update operation'''
        input_data = self._input_nodes[0].get_data()
        gradient_data = self._output_nodes[0].get_grad()
        self._updator.cal_update_values(input_data, gradient_data)
        self._updator.do_update(self._layer._weights)

    def save_history(self, state_node):
        '''save history state'''
        self._hist_nodes.insert(0, self._hist_nodes.pop())    
        t_node = self._hist_nodes[0]
        opterations.np_copy_nodes_data(state_node, t_node)
        if self._hist_index < self._bptt+1:
            self._hist_index += 1

    def update_history(self, gradient_node):
        '''bptt training via history states'''
        for i in xrange(self._hist_index-1):
            input_data = self._hist_nodes[i+1].get_data()
            gradient_data = self._hist_nodes[i].get_grad()
            self._updator.cal_update_values(input_data, gradient_data)
            self._updator.do_update(self._layer._weights)
            #cal 


    def _create_hist_nodes(self, batch_size, data_size, gradient_data, hist_size):
        # all hist node share the same gradient_data with output node
        hist_nodes = [node.Node(util.init_np_zeros_weights(batch_size, data_size), grad=gradient_data) for i in xrange(hist_size)]
        return hist_nodes
        