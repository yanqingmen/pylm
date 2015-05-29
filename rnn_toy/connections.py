'''
connections for rnn
author: hzx
'''
import layer
import operations
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
        '''
        check input data dimension
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
        self._input_nodes = [prev_conn.get_output_nodes()[0]]

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
        '''backprob operation'''
        pass


class RecurrentConnect(object):
    '''connection based on recurrent layer'''
    def __init__(self, hiden_size, batch_size, alpha=0.01, bptt=5):
        self._layer = layer.RecurrentLayer(hiden_size,\
         operations.np_sigmoid_op, operations.np_sigmoid_back_op)
        self._updator = updator.WeightsUpdator(hiden_size, hiden_size, alpha)
        self._hiden_size = hiden_size
        self._batch_size = batch_size
        # recurrent layer has a defualt data node which shared by input and output
        # self._input_nodes = [util.create_np_node(batch_size, hiden_size, init_random=True)]
        self._input_nodes = [util.create_np_node(batch_size, hiden_size)]
        # besides the default data node, output nodes also include a tmp data node for save tmp data
        self._output_nodes = [self._input_nodes[0], util.create_np_node(batch_size, hiden_size)]
        self._bptt = bptt
        # shared gradient data
        gradient_data = self._output_nodes[1].get_grad()
        self._hist_nodes = self._create_hist_nodes(batch_size, hiden_size, bptt, gradient_data)
        self._hist_index = 1

    def check_data_dimension(self):
        '''check the input data dimension'''
        for tnode in self._input_nodes[1:]:
            tshape = tnode.get_data().shape
            if not tshape == (self._batch_size, self._hiden_size):
                raise TypeError("shape of inputdata is not fit with this connect,(%s,%s):(%s,%s)",\
             (tshape[0], tshape[1], self._batch_size, self._hiden_size))

    def set_input_nodes(self, input_nodes):
        '''add additional input_nodes to this connect'''
        for input_node in input_nodes:
            self._input_nodes.append(input_node)

    def link_to_previous(self, prev_conn):
        '''link this connection to previous connection'''
        self._input_nodes.append(prev_conn.get_output_nodes()[0])

    def get_output_nodes(self):
        '''get output nodes of this connection'''
        return [self._output_nodes[0]]

    def forward(self):
        '''forward operation'''
        self._layer.forward(self._input_nodes, self._output_nodes)
        self.save_history(self._output_nodes[0])

    def backprob(self):
        '''backprob operation'''
        # operations.np_copy_nodes_grad(self._output_nodes[0], self._hist_nodes[0])
        self._layer.backprob(self._input_nodes, self._output_nodes)

    def update(self):
        '''update operation'''
        self.update_history()

    def save_history(self, state_node):
        '''save history state'''
        self._hist_nodes.insert(0, self._hist_nodes.pop())
        t_node = self._hist_nodes[0]
        operations.np_copy_nodes_data(state_node, t_node)
        if self._hist_index < self._bptt+1:
            self._hist_index += 1

    def reset_state(self):
        '''reset_state and history'''
        self._hist_index = 1
        self._input_nodes[0].get_data()[:] = 0
        self._hist_nodes[0].get_data()[:] = 0

    def update_history(self):
        '''bptt training via history states'''
        for i in xrange(self._hist_index-1):
            input_data = self._hist_nodes[i+1].get_data()
            gradient_data = self._hist_nodes[i].get_grad()
            self._updator.cal_update_values(input_data, gradient_data)
            self._updator.do_update(self._layer.get_weights())
            #cal new gradient
            c_input_nodes = [self._hist_nodes[i+1]]
            c_output_nodes = [self._hist_nodes[i], self._output_nodes[1]]
            self._layer.backprob(c_input_nodes, c_output_nodes)

    def _create_hist_nodes(self, batch_size, data_size, hist_size, gradient_data):
        '''create history state nodes'''
        # all hist node share the same gradient_data
        hist_nodes = [util.create_np_node(batch_size, data_size, grad=gradient_data) for i in xrange(hist_size+1)]
        operations.np_copy_nodes_data(self._input_nodes[0], hist_nodes[0])
        return hist_nodes


class FullCLayerConnect(object):
    '''fullconnlayer based connection'''
    def __init__(self, input_size, output_size, batch_size, alpha=0.01, no_bias=False):
        self._input_size = input_size
        self._output_size = output_size
        self._batch_size = batch_size
        self._layer = layer.FullConnectLayer(input_size, output_size, no_bias=no_bias)
        self._updator = updator.WeightsUpdator(input_size, output_size, alpha)
        if no_bias is False:
            self._bias_updator = updator.BiasUpdator(output_size, alpha)
        else:
            self._bias_updator = None
        self._output_nodes = [util.create_np_node(batch_size, output_size)]
        self._input_nodes = None

    def check_data_dimension(self):
        '''check the input data dimension'''
        input_node = self._input_nodes[0]
        tshape = input_node.get_data().shape
        if not tshape == (self._batch_size, self._input_size):
            raise TypeError("shape of inputdata is not fit with this connect,(%s,%s):(%s,%s)",\
             (tshape[0], tshape[1], self._batch_size, self._input_size))

    def set_input_nodes(self, input_nodes):
        '''set input nodes for this connection'''
        self._input_nodes = input_nodes

    def link_to_previous(self, prev_conn):
        '''link this connection to previous connection'''
        self._input_nodes = [prev_conn.get_output_nodes()[0]]

    def get_output_nodes(self):
        '''get output nodes of this connection'''
        return [self._output_nodes[0]]

    def forward(self):
        '''forward operation'''
        self._layer.forward(self._input_nodes, self._output_nodes)

    def backprob(self):
        '''backprob operation'''
        self._layer.backprob(self._input_nodes, self._output_nodes)

    def update(self):
        '''update weights'''
        input_data = self._input_nodes[0].get_data()
        gradient_data = self._output_nodes[0].get_grad()
        self._updator.cal_update_values(input_data, gradient_data)
        self._updator.do_update(self._layer.get_weights())
        if self._bias_updator is not None:
            self._bias_updator.cal_update_values(gradient_data)
            self._bias_updator.do_update(self._layer.get_bias())


class SoftMaxConnect(object):
    '''connect based on softmax'''
    def __init__(self):
        self._output_nodes = None
        self._target_data = None

    def link_to_previous(self, prev_conn):
        '''link with previous connection'''
        self._output_nodes = [prev_conn.get_output_nodes()[0]]

    def set_target(self, target_data):
        '''set target data'''
        self._target_data = target_data

    def get_output_nodes(self):
        '''get output nodes of this connection'''
        return [self._output_nodes[0]]

    def forward(self):
        '''forward operation'''
        operations.np_softmax_op(self._output_nodes[0])

    def backprob(self):
        '''backprob operation'''
        output_data = self._output_nodes[0].get_data()
        gradient_data = self._output_nodes[0].get_grad()
        util.np_cal_gradient(output_data, self._target_data, gradient_data)

    def update(self):
        '''no update need be done for softmax connect'''
        pass
