#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell

class Attention_sum_reader(object):
    def __init__(self, d_len, q_len, A_len, lr, embedding_matrix, hidden_size, num_layers):
        self._d_len = d_len
        self._q_len = q_len
        self._A_len = A_len
        self._lr = lr
        self._embedding_matrix = embedding_matrix
        self._hidden_size = hidden_size
        self._num_layers = num_layers

        self._q_input = tf.placeholder(dtype=tf.int32, shape=(None, q_len), name='q_input')
        self._d_input = tf.placeholder(dtype=tf.int32, shape=(None, d_len), name='d_input')
        self._context_mask = tf.placeholder(dtype=tf.int8, shape=(None, d_len), name='context_mask')
        self._A_mask = tf.placeholder(dtype=tf.int8, shape=(None, A_len, d_len), name='A_mask')
        self._y = tf.placeholder(dtype=tf.int32, shape=(None), name='y')

        self._build_network()

    def train(self, sess, provider):
        optimizer = self._Optimizer()
        train_op = optimizer.minimize(self._total_loss)

        sess.run(tf.global_variables_initializer())
        print('pig')
        for (step_count, data) in enumerate(provider):
            q_input, d_input, context_mask, A_mask, y = data
            print('miao', len(q_input))
            _, total_loss = sess.run([train_op, sess._total_loss], feed_dict={
                    self._q_input: q_input,
                    self._d_input: d_input,
                    self._context_mask: context_mask,
                    self._A_mask: A_mask,
                    self._y: y})
            print('wang')
            if step_count % 10 == 0:
                logging.info('[Train] step_count: {}, loss: {}'.format(step_count + 1, loss))

    def test(self):
        pass

    def _RNNCell(self):
        return LSTMCell(self._hidden_size)

    def _MultiRNNCell(self):
        return MultiRNNCell([self._RNNCell() for _ in range(self._num_layers)])

    def _Optimizer(self):
        return tf.train.AdamOptimizer(self._lr)

    def _build_network(self):
        with tf.variable_scope('q_encoder'):
            q_embed = tf.nn.embedding_lookup(self._embedding_matrix, self._q_input)
            q_lens = tf.reduce_sum(tf.sign(tf.abs(self._q_input)), 1)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_bw=self._MultiRNNCell(), cell_fw=self._MultiRNNCell(),
                    inputs=q_embed, dtype=tf.float32, sequence_length=q_lens)
            q_encode = tf.concat([final_states[0][-1][1], final_states[1][-1][1]], axis=-1)

            # [batch_size, hidden_size * 2]
            logging.info('q_encode shape {}'.format(q_encode.get_shape()))

        with tf.variable_scope('d_encoder'):
            d_embed = tf.nn.embedding_lookup(self._embedding_matrix, self._d_input)
            d_lens = tf.reduce_sum(tf.sign(tf.abs(self._d_input)), 1)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_bw=self._MultiRNNCell(), cell_fw=self._MultiRNNCell(),
                    inputs=d_embed, dtype=tf.float32, sequence_length=d_lens)
            d_encode = tf.concat(outputs, axis=-1)

            # [batch_size, d_len, hidden_size * 2]
            logging.info('d_encode shape {}'.format(d_encode.get_shape()))

        with tf.variable_scope('dot_sum'):
            attention_value = tf.map_fn(
                    lambda v: tf.reduce_sum(tf.multiply(v[0], v[1]), -1),
                    (q_encode, d_encode),
                    dtype=tf.float32)
            attention_value_masked = tf.multiply(attention_value, tf.cast(self._context_mask, tf.float32))
            attention_value_softmax = tf.nn.softmax(attention_value_masked)
            attention_sum = tf.map_fn(
                    lambda v: tf.reduce_sum(tf.multiply(v[0], v[1]), -1),
                    (attention_value_softmax, tf.cast(self._A_mask, tf.float32)), 
                    dtype=tf.float32)
            
            # [batch_size, A_len]
            logging.info('attention_sum shape {}'.format(attention_sum.get_shape()))

        with tf.variable_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._y, logits=attention_sum)
            self._total_loss = tf.reduce_mean(losses)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    embedded = tf.zeros((1000, 100), dtype=tf.float32)
    Attention_sum_reader(d_len=600, q_len=60, A_len=10, lr=0.1, embedding_matrix=embedded, hidden_size=128, num_layers=2)
