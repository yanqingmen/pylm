'''
operations for forward or backprob
author: hzx
'''
import numpy as np
import util


def np_emb_op(weights, input_node, output_node):
    '''embedding operation for numpy matrix'''
    input_data = input_node.get_data()
    output_data = output_node.get_data()
    output_data[:] = weights[input_data]

def np_full_connect_op(weights, bias, input_node, output_node):
    '''full connect layer operation for numpy matrix'''
    input_data = input_node.get_data()
    output_data = output_node.get_data()
    np.dot(input_data, weights, output_data)
    if bias is not None:
        output_data += bias

def np_full_connect_back_op(weights, bias, input_node, output_node):
    '''back prob operation for full connect layer'''
    input_data = input_node.get_grad()
    output_data = output_node.get_grad()
    np.dot(input_data, weights, output_data)
    if bias is not None:
        output_data += bias

def np_sigmoid_op(input_node, output_node):
    '''sigmoid activation operation for numpy matrix'''
    input_data = input_node.get_data()
    output_data = output_node.get_data()
    np.copyto(output_data, input_data)
    util.np_sigmoid(output_data)

def np_sigmoid_back_op(input_node, output_node):
    '''back prob operation of sigmoid activation for numpy matrix'''
    grad_data = input_node.get_grad()
    state_data = input_node.get_data()
    output_data = output_node.get_grad()
    util.np_sigmoid_backprob(grad_data, state_data, output_data)

def np_add_nodes_data(input_node, output_node):
    '''add data of input_node to data of output_node for numpy matrix'''
    input_data = input_node.get_data()
    output_data = output_node.get_data()
    output_data += input_data

def np_copy_nodes_data(input_node, output_node):
    '''copy data of input_node to data of output_node'''
    input_data = input_node.get_data()
    output_data = output_node.get_data()
    np.copyto(output_data, input_data)

def np_copy_nodes_grad(input_node, output_node):
    '''copy grad of input_node to grad of output_node'''
    input_data = input_node.get_grad()
    output_data = output_node.get_grad()
    np.copyto(output_data, input_data)
