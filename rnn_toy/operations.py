'''
operations for forward or backprob
author: hzx
'''
import numpy as np


def np_emb_op(weights, input_nodes, output_nodes):
    '''embedding operation for numpy matrix'''
    input_data = input_nodes[0].get_data()
    output_data = output_nodes[0].get_data()
    output_data[:] = weights[input_data]

def np_full_connect_op(weights, bias, input_nodes, output_nodes):
    '''full connect layer operation for numpy matrix'''
    input_data = input_nodes[0].get_data()
    output_data = output_nodes[0].get_data()
    np.dot(input_data, weights, output_data)
    if bias is not None:
        output_data += bias

def np_recurrent_op(weights, bias, input_nodes, output_nodes):
    '''recurrent layer operation for numpy matrix'''
    f_input_node = input_nodes[0]
    s_input_node = input_nodes[1]
    output_node = output_nodes[0]
    output_node_tmp = output_nodes[1]
    np_full_connect_op(weights, bias, [f_input_node], [output_node])
    np_full_connect_op(weights, bias, [s_input_node], [output_node_tmp])
    output_data = output_node.get_data()
    output_data += output_node_tmp.get_data()
    output_data_tmp = output_node_tmp.get_data()
    output_data_tmp[:] = output_data

def np_recurrent_back_op(weights, input_nodes, output_nodes):
    '''recurrent layer backprob operation for numpy matrix'''
    np_full_connect_op(weights, None, input_nodes, output_nodes)
    f_output_data = output_nodes[0].get_data()
    s_output_data = output_nodes[1].get_data()
    s_output_data[:] = f_output_data
