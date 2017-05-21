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

os.environ["CUDA_VISIBLE_DEVICES"]="2"
logging.basicConfig(level=logging.INFO)
random.seed(1)

d_len = 1000
q_len = 150
A_len = 10
lr_init = 0.005
lr_decay = 2000
hidden_size = 128
batch_size = 128
step_num = 20000
embed_dim = 100
embed_file = '/data1/flashlin/data/glove/glove.6B.100d.txt'

word_dict = load_vocab('out/vocab')
embedding_matrix = gen_embeddings(word_dict, embed_dim, embed_file)
embedding_matrix = embedding_matrix.astype('float32')
asr = Attention_sum_reader('pig', d_len, q_len, A_len, lr_init, lr_decay, 
        embedding_matrix, hidden_size)

#src_data = read_cbt_data('out/cbtest_NE_train.txt.idx', [100, d_len], [10, q_len])
#provider = data_provider(src_data, batch_size, d_len, q_len, step_num=step_num)

src_data = read_cbt_data('out/cbtest_NE_valid_2000ex.txt.idx', [100, d_len], [10, q_len])
provider = data_provider(src_data, batch_size, d_len, q_len, None, 1)

with tf.Session() as sess:
    #asr.train(sess, provider, 'pig', 500)
    asr.test(sess, provider, 'gru_model/gru-19800')
