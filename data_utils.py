#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import random
import re
import logging
import nltk
import numpy as np
import tensorflow as tf

from collections import Counter

_PAD = u'_PAD'
_UNK = u'_UNK'
_EOS = u'_EOS'
_START_VOCAB = [_PAD, _UNK, _EOS]

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2

def writeWrapper(fh, context):
    if not type(context) is unicode:
        assert(type(context) is str)
        context = unicode(context, 'utf8')
    fh.write(context)

def tokenizer(sentence):
    sentence = ' '.join(sentence.split("|"))
    tokens = nltk.word_tokenize(sentence.lower())
    return tokens

def gen_vocab(data_file, word_dict=None):
    word_dict = word_dict if word_dict else Counter()

    for i, line in enumerate(io.open(data_file, 'r', encoding='utf8')):
        if len(line.strip()) == 0:
            continue

        tokens = tokenizer(line[line.index(' ') + 1:])
        word_dict.update(tokens)

        if i % 100000 == 0:
            logging.info('[gen_vocab] data_file: %s, i: %d' % (data_file, i))

    return word_dict

def save_vocab(word_dict, vocab_file):
    with io.open(vocab_file, 'w', encoding='utf8') as f:
        for word in _START_VOCAB:
            writeWrapper(f, word + u'\n')
        for word in word_dict:
            writeWrapper(f, word + u'\n')

def load_vocab(vocab_file):
    assert(vocab_file != None)
    word_dict = {}
    with io.open(vocab_file, 'r', encoding='utf8') as f:
        for wid, word in enumerate(f):
            word = word.strip()
            word_dict[word] = wid
    return word_dict

def sentence_to_token_ids(sentence, word_dict):
    return [word_dict.get(token, UNK_ID) for token in tokenizer(sentence)]

def cbt_data_to_token_ids(data_file, target_file, vocab_file):
    word_dict = load_vocab(vocab_file)

    with io.open(data_file, 'r', encoding='utf8') as data_file, io.open(target_file, 'w', encoding='utf8') as tokens_file:
        for line in data_file:
            if len(line.strip()) == 0:
                writeWrapper(tokens_file, u'\n')
                continue

            num = int(line[:line.index(' ')])
            line = line[line.index(' ') + 1:]

            if num == 21:
                q, a, _, A = line.split('\t')
                tokens_ids_q = sentence_to_token_ids(q, word_dict)
                tokens_ids_A = [word_dict.get(Ai.lower(), UNK_ID) for Ai in A.rstrip('\n').split('|')]
                context = (' '.join([str(token) for token in tokens_ids_q]) + '\t' +
                        str(word_dict.get(a.lower(), UNK_ID)) + '\t' +
                        '|'.join([str(token) for token in tokens_ids_A]) + '\n')
                context = unicode(context, 'utf8')
                writeWrapper(tokens_file, context)
            else:
                tokens_ids = sentence_to_token_ids(line, word_dict)
                context = ' '.join([str(token) for token in tokens_ids]) + '\n'
                writeWrapper(tokens_file, context)

def prepare_cbt_data(data_dir, out_dir, train_file, valid_file, test_file):
    if not tf.gfile.Exists(out_dir):
        os.mkdir(out_dir)

    src_train_file = os.path.join(data_dir, train_file)
    src_valid_file = os.path.join(data_dir, valid_file)
    src_test_file = os.path.join(data_dir, test_file)
    idx_train_file = os.path.join(out_dir, train_file + ".idx")
    idx_valid_file = os.path.join(out_dir, valid_file + ".idx")
    idx_test_file = os.path.join(out_dir, test_file + ".idx")
    vocab_file = os.path.join(out_dir, "vocab")

    wd = gen_vocab(src_train_file)
    wd = gen_vocab(src_valid_file, wd)
    wd = gen_vocab(src_test_file, wd)
    save_vocab(wd, vocab_file)
    logging.info('Total words: %d' % len(wd))
    logging.info('Total distinct words: %d' % sum(wd.values()))

    cbt_data_to_token_ids(src_train_file, idx_train_file, vocab_file)
    cbt_data_to_token_ids(src_valid_file, idx_valid_file, vocab_file)
    cbt_data_to_token_ids(src_test_file, idx_test_file, vocab_file)

    return idx_train_file, idx_valid_file, idx_train_file, vocab_file

def read_cbt_data(idx_file, d_len_range = None, q_len_range = None, max_count = None):
    def ok(d_len, q_len, A_len):
        d_con = (not d_len_range) or (d_len_range[0] < d_len < d_len_range[1])
        q_con = (not q_len_range) or (q_len_range[0] < q_len < q_len_range[1])
        A_con = (A_len == 10)
        return d_con and q_con and A_con

    skip = 0
    documents, questions, answers, candidates = [], [], [], []
    with io.open(idx_file, 'r', encoding='utf8') as f:
        cnt = 0
        d, q, a, A = [], [], [], []
        for line in f:
            cnt += 1
            if cnt <= 20:
                d.extend(line.strip().split(' ') + [EOS_ID])
            elif cnt == 21:
                tmp = line.strip().split('\t')
                q = tmp[0].split(' ') + [EOS_ID]
                a = [1 if tmp[1] == wid else 0 for wid in d]
                A = [Ai for Ai in tmp[2].split('|')]
                A.remove(tmp[1])
                A.insert(0, tmp[1])

                if ok(len(d), len(q), len(A)):
                    documents.append(d)
                    questions.append(q)
                    answers.append(a)
                    candidates.append(A)
                else:
                    skip += 1
            elif cnt == 22:
                d, q, a, A = [], [], [], []
                cnt = 0

            if max_count and len(questions) >= max_count:
                break; 

    logging.info('[read_cbt_data] skip: {}, read: {}'.format(skip, len(questions)))

    return documents, questions, answers, candidates

def get_embed_dim(embed_file):
    line = io.open(embed_file, 'r', encoding='utf8').readline()
    return len(line.split()) - 1

def gen_embeddings(word_dict, embed_dim, embed_file=None):
    num_words = len(word_dict)
    #return tf.random_uniform([num_words, embed_dim], -0.1, 0.1)
    
    embedding_matrix = np.random.uniform(-0.1, 0.1, [num_words, embed_dim])
    if embed_file:
        pre_trained = 0
        for line in io.open(embed_file, 'r', encoding='utf8'):
            items = line.split()
            word = items[0]
            assert(embed_dim + 1 == len(items))
            if word in word_dict:
                pre_trained += 1
                embedding_matrix[word_dict[word]] = [float(x) for x in items[1:]]
        
        logging.info('Embedding file: %s, pre_trained_rate: %.2f' % (embed_file, 100.0 * pre_trained / num_words))
    
    return embedding_matrix

def data_provider(src_data, batch_size, d_len, q_len, step_num, epoch_num):
    documents, questions, answers, candidates = src_data
    N = len(documents)
    
    logging.info('[data_provider] N: {}, batch_size: {}, step_num: {}, d_len: {}, q_len: {}'.format(
                N, batch_size, step_num, d_len, q_len))
    assert(len(questions) == N and len(answers) == N and len(candidates) == N)
    assert(N > batch_size * 10)

    context_masks = []
    ys = []
    for i in range(N):
        context_mask = [1] * len(documents[i]) + [0] * (d_len - len(documents[i]))
        context_masks.append(context_mask)

        assert(len(documents[i]) <= d_len)
        documents[i] += [PAD_ID] * (d_len - len(documents[i]))

        assert(len(questions[i]) <= q_len)
        questions[i] += [PAD_ID] * (q_len - len(questions[i]))

        ys.append(0)

    if not step_num:
        assert(epoch_num)
        step_num = N // batch_size
        if N % batch_size != 0:
            step_num += 1
        step_num *= epoch_num

    h = N
    idx = [i for i in range(N)]
    for _ in range(step_num):
        if h == N:
            random.shuffle(idx)
            h = 0
            logging.info('[data_provider] new epoch')

        d_input = []
        q_input = []
        context_mask = []
        ca = []
        y = []

        data_len = batch_size if h + batch_size <= N else N - h
        for i in range(data_len):
            d_input.append(documents[idx[h + i]])
            q_input.append(questions[idx[h + i]])
            context_mask.append(context_masks[idx[h + i]])
            ca.append(candidates[idx[h + i]])
            y.append(ys[idx[h + i]])
        h += data_len

        yield d_input, q_input, context_mask, ca, y

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    idx_train, idx_valid, idx_test, vocab = prepare_cbt_data(
            '/data1/flashlin/data/CBTest/data/', 'out', 'cbtest_NE_train.txt', 'cbtest_NE_valid_2000ex.txt', 'cbtest_NE_test_2500ex.txt')

