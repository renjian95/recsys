#!/usr/bin/env python
# encoding: utf-8
'''
@author: renjian
@contact: 1838915959@qq.com
@file: tfrecords_op.py
@time: 2019-05-07 17:00
@desc:
'''

import tensorflow as tf
import os


def train_input_fn(input, batch_size, shuffle_size, parse_example):

    if isinstance(input, str) and os.path.isdir(input):
        train_files = [input + '/' + x for x in os.listdir(input)]
    else:
        train_files = input

    files = tf.data.Dataset.list_files(train_files)
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=5))
    # 样本解析, shuffle, 设置batch_size
    if shuffle_size > 0:
        dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.map(parse_example, num_parallel_calls=8)
    dataset = dataset.repeat().batch(batch_size).prefetch(1)
    print(dataset.output_types)
    print(dataset.output_shapes)
    return dataset


def eval_input_fn(filename, batch_size, parse_example):
    if isinstance(filename, str) and os.path.isdir(filename):
        eval_files = [filename + '/' + x for x in os.listdir(filename)]
    else:
        eval_files = filename

    dataset = tf.data.TFRecordDataset(eval_files)
    dataset = dataset.map(parse_example, num_parallel_calls=8)
    # 样本解析, shuffle, 设置batch_size
    dataset = dataset.batch(batch_size)
    return dataset