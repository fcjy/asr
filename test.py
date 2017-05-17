#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import tensorflow as tf

from attention_sum_reader import Attention_sum_reader
from data_utils import load_vocab, gen_embeddings, read_cbt_data, data_provider

os.environ["CUDA_VISIBLE_DEVICES"]="3"
logging.basicConfig(level=logging.INFO)
random.seed(1)

d_len = 1000
q_len = 150
A_len = 10
lr = 0.5
hidden_size = 128
num_layers = 1
batch_size = 100
step_num = 10000
embed_dim = 100
embed_file = '/data1/flashlin/data/glove/glove.6B.100d.txt'

word_dict = load_vocab('out/vocab')
embedding_matrix = gen_embeddings(word_dict, embed_dim, embed_file)
embedding_matrix = embedding_matrix.astype('float32')
#embedding_matrix = tf.convert_to_tensor(embedding_matrix)
asr = Attention_sum_reader(d_len, q_len, A_len, lr, embedding_matrix, hidden_size, num_layers)
src_data = read_cbt_data('out/cbtest_NE_train.txt.idx', [100, d_len], [10, q_len])
provider = data_provider(src_data, batch_size, step_num, d_len, q_len)

with tf.Session() as sess:
    asr.train(sess, provider)
