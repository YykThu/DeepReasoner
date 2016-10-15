import tensorflow as tf
import numpy as np


def simple_dense_layer(input, flag):
    n_in = flag.dense_n_in
    n_out = flag.dense_n_out
    with tf.name_scope('dense_layer') as scope:
        weight = tf.Variable(initial_value=tf.random_normal(shape=[n_in, n_out],
                                                            mean=0,
                                                            stddev=0.1,
                                                            dtype=tf.float32),
                             name='weight')
        bias = tf.Variable(initial_value=tf.constant(value=0,
                                                     shape=[n_out, ],
                                                     dtype=tf.float32),
                           name='bias')
        energy = tf.matmul(input, weight) + bias
        output = tf.sigmoid(energy, name='dense_output')
    return output


def loss_dense_layer(dense_output, targets):
    loss = tf.reduce_mean(tf.square(dense_output - targets))
    return loss


def softmax_layer(feature, flag):
    n_in = flag.softmax_n_in
    n_class = flag.n_class
    with tf.name_scope('softmax') as scope:
        weight = tf.Variable(initial_value=tf.random_normal(shape=[n_in, n_class],
                                                            mean=0,
                                                            stddev=0.1,
                                                            dtype=tf.float32),
                             name='weight')
        bias = tf.Variable(initial_value=tf.constant(value=0,
                                                     shape=[n_class, ],
                                                     dtype=tf.float32),
                           name='bias')
        logits = tf.matmul(feature, weight) + bias
        pre_class = tf.nn.softmax(logits)
    return logits, pre_class


def loss_softmax_layer(logits, label):
    loss = tf.reduce_mean(tf.sparse_softmax(logits, label))
    return loss


def lstm(feature, flag):
    n_in = flag.lstm_n_in
    n_out = flag.lstm_n_out
    dim_cell = flag.lstm_dim_cell
    dim_input_gate = flag.lstm_dim_input_gate
    dim_forget_gate = flag.lstm_dim_forget_gate
    dim_output_gate = flag.lstm_dim_output_gate
    with tf.name_scope('lstm'):
        weight = tf.Variable(initial_value=tf.random_normal(shape=[n_in + dim_cell, dim_cell],
                                                            mean=0,
                                                            stddev=0.1,
                                                            dtype=tf.float32),
                             name='weight')
        bias = tf.Variable(initial_value=tf.constant(value=0,
                                                     shape=[dim_cell, ],
                                                     dtype=tf.float32))
        weight_ig = tf.Variable(initial_value=tf.random_normal(shape=[n_in + dim_cell, dim_input_gate],
                                                               mean=0,
                                                               stddev=0.1,
                                                               dtype=tf.float32),
                                name='weight_input_gate')
        bias_ig = tf.Variable(initial_value=tf.constant(value=0,
                                                        shape=[dim_input_gate, ],
                                                        dtype=tf.float32),
                              name='bias_input_gate')
        weight_fg = tf.Variable(initial_value=tf.random_normal(shape=[n_in + dim_cell, dim_forget_gate],
                                                               mean=0,
                                                               stddev=0.1,
                                                               dtype=tf.float32),
                                name='weight_forget_gate')
        bias_fg = tf.Variable(initial_value=tf.constant(value=0,
                                                        shape=[dim_forget_gate, ],
                                                        dtype=tf.float32),
                              name='bias_forget_gate')
        weight_og = tf.Variable(initial_value=tf.random_normal(shape=[n_in + dim_cell, dim_output_gate],
                                                               mean=0,
                                                               stddev=0.1,
                                                               dtype=tf.float32),
                                name='weight_output_gate')
        bias_og = tf.Variable(initial_value=tf.constant(value=0,
                                                        shape=[dim_output_gate, ],
                                                        dtype=tf.float32),
                              name='bias_output_gate')
        cell = tf.Variable(initial_value=tf.constant(value=0,
                                                     shape=[dim_cell, ],
                                                     dtype=tf.float32),
                           name='memory_cell')
        hidden = tf.Variable(initial_value=tf.constant(value=0,
                                                       shape=[n_out, ],
                                                       dtype=tf.float32))
        input_gate = tf.sigmoid(tf.matmul([feature, hidden], weight_ig) + bias_ig)
        forget_gate = tf.sigmoid(tf.matmul([feature, hidden], weight_fg) + bias_fg)
        output_gate = tf.sigmoid(tf.matmul([feature, hidden], weight_og) + bias_og)
        input_feature = tf.tanh(tf.matmul([feature, hidden], weight) + bias)
        cell =  

        return output