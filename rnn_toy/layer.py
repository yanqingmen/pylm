'''
layers for recurrent neural networks
'''
import util
import operations

class EmbdLayer(object):
    '''embedding layer'''
    def __init__(self, input_size, output_size):
        self._weights = util.init_np_weights(input_size, output_size)

    def forward(self, input_nodes, output_nodes):
        '''
        forward operation
        * layer object will not check the dimension of inputdata and outputdata, 
        * which will be done by connection object
        '''
        if not len(input_nodes) == len(output_nodes) == 1:
            raise TypeError("EmbdLayer can only handle one-to-one node pair")
        operations.np_emb_op(self._weights, input_nodes, output_nodes)

    def backprob(self, input_nodes, output_nodes):
        '''
        backward operation
        * layer object will not check the dimension of inputdata and outputdata, 
        * which will be done by connection object
        '''
        raise NotImplementedError("EmbdLayer has no backprob operation")


class FullConnectLayer(object):
    '''normal full connection layer'''
    def __init__(self, input_size, output_size):
        self._weights = util.init_np_weights(input_size, output_size)
        self._bias = util.init_np_weights(1, output_size)

    def forward(self, input_nodes, output_nodes):
        '''
        forward operation
        * layer object will not check the dimension of inputdata and outputdata, 
        * which will be done by connection object
        '''
        if not len(input_nodes) == len(output_nodes) == 1:
            raise TypeError("full connection layer can only handle one-to-one node pair")
        operations.np_full_connect_op(self._weights, self._bias, input_nodes, output_nodes)

    def backprob(self, input_nodes, output_nodes):
        '''
        backward operation
        * layer object will not check the dimension of inputdata and outputdata, 
        * which will be done by connection object
        '''
        if not len(input_nodes) == len(output_nodes) == 1:
            raise TypeError("full connection layer can only handle one-to-one node pair")
        operations.np_full_connect_op(self._weights.T, None, output_nodes, input_nodes)


class RecurrentLayer(object):
    '''recurrent layer'''
    def __init__(self, input_size, output_size):
        self._weights = util.init_np_weights(input_size, output_size)
        self._bias = util.init_np_weights(1, output_size)

    def forward(self, input_nodes, output_nodes):
        '''
        forward operation
        * layer object will not check the dimension of inputdata and outputdata, 
        * which will be done by connection object
        '''
        if not len(input_nodes) == len(output_nodes) == 2:
            raise TypeError("RecurrentLayer can only handle two-to-two node pair,\
             (one output node for save tmp data)")
        operations.np_recurrent_op(self._weights, self._bias, input_nodes, output_nodes)

    def backprob(self, input_nodes, output_nodes):
        '''
        backward operation
        * layer object will not check the dimension of inputdata and outputdata, 
        * which will be done by connection object
        '''
        if not len(input_nodes) == len(output_nodes) == 2:
            raise TypeError("RecurrentLayer can only handle two-to-two node pair,\
             (the second output node is used for saving tmp data and code facility)")
        operations.np_recurrent_back_op(self._weights.T, output_nodes, input_nodes)
