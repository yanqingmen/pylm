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
        operations.np_emb_op(self._weights, input_nodes[0], output_nodes[0])

    def backprob(self, input_nodes, output_nodes):
        '''
        backward operation
        * layer object will not check the dimension of inputdata and outputdata, 
        * which will be done by connection object
        '''
        raise NotImplementedError("EmbdLayer has no backprob operation")


class FullConnectLayer(object):
    '''normal full connection layer'''
    def __init__(self, input_size, output_size, no_bias=False):
        self._weights = util.init_np_weights(input_size, output_size)
        if no_bias:
            self._bias = None
        else:
            self._bias = util.init_np_weights(1, output_size)

    def forward(self, input_nodes, output_nodes):
        '''
        forward operation
        * layer object will not check the dimension of inputdata and outputdata, 
        * which will be done by connection object
        '''
        if not len(input_nodes) == len(output_nodes) == 1:
            raise TypeError("full connection layer can only handle one-to-one node pair")
        operations.np_full_connect_op(self._weights, self._bias, input_nodes[0], output_nodes[0])

    def backprob(self, input_nodes, output_nodes):
        '''
        backward operation
        * layer object will not check the dimension of inputdata and outputdata, 
        * which will be done by connection object
        '''
        if not len(input_nodes) == len(output_nodes) == 1:
            raise TypeError("full connection layer can only handle one-to-one node pair")
        operations.np_full_connect_op(self._weights.T, None, output_nodes[0], input_nodes[0])


class ActivationLayer(object):
    '''activation layer'''
    def __init__(self, forward_operation, backprob_operation):
        self._forward = forward_operation
        self._backprob = backprob_operation

    def forward(self, input_nodes, output_nodes):
        '''
        forward operation
        '''
        if not len(input_nodes) == len(output_nodes) == 1:
            raise TypeError("ActivationLayer can only handle one-to-one node pair")
        self._forward(input_nodes[0], output_nodes[0])

    def backprob(self, input_nodes, output_nodes):
        '''
        backprob operation
        '''
        if not len(input_nodes) == len(output_nodes) == 1:
            raise TypeError("ActivationLayer can only handle one-to-one node pair")
        self._backprob(output_nodes[0], input_nodes[0])



class RecurrentLayer(object):
    '''recurrent layer, which contains a fullconnect layer and an activation layer'''
    def __init__(self, hiden_size, forward_operation, backprob_operation):
        self._f_layer = FullConnectLayer(hiden_size, hiden_size, True)
        self._act_layer = ActivationLayer(forward_operation, backprob_operation)

    def forward(self, input_nodes, output_nodes):
        '''forward operation'''
        if not len(input_nodes) >= 2 and len(output_nodes) == 2:
            raise TypeError("recurrent layer can only handle n-to-two node pair,\
                the seconde output_node is used for save tmp data")
        self._f_layer.forward([input_nodes[0]], [output_nodes[1]])
        for node in input_nodes[1:]:
            operations.np_add_nodes(node, output_nodes[1])
        self._act_layer.forward([output_nodes[1]], [output_nodes[0]])

    def backprob(self, input_nodes, output_nodes):
        '''backprob operation'''
        if not len(input_nodes) >= 1 and len(output_nodes) == 2:
            raise TypeError("recurrent layer can only handle n-to-two node pair,\
                the seconde output_node is used for save tmp data")
        self._act_layer.backprob([output_nodes[0]], [output_nodes[1]])
        self._f_layer.backprob([output_nodes[1]], [input_nodes[0]])
        for node in input_nodes[1:]:
            operations.np_copy_nodes(input_nodes[0], node)


class LossLayer(object):
    '''loss layer'''
    def __init__(self, loss_operation):
        self._forward = loss_operation

    def forward(self, input_nodes, output_nodes):
        '''
        forward operation
        '''
        if not len(input_nodes) == len(output_nodes) == 1:
            raise TypeError("LossLayer can only handle one-to-one node pair")
        self._forward(input_nodes[0], output_nodes[0])

    def backprob(self, input_nodes, output_nodes):
        '''Loss layer would do nothing for backprob'''
        pass
