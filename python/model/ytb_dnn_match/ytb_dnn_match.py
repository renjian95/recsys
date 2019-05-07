#!/usr/bin/env python
# encoding: utf-8
'''
@author: renjian
@contact: 1838915959@qq.com
@file: ytb_dnn_match.py
@time: 2018-11-02 21:43
@desc:
'''

import tensorflow as tf
from tensorflow import estimator
from tensorflow.python.estimator.canned import optimizers
from tensorflow import nn
from model.utils.tfrecords_op import *
import math
import os
from tensorflow import feature_column as fc
from tensorflow.python.feature_column.feature_column_v2 import EmbeddingColumn


def build_mode_norm(features, mode, params):
    # Build the hidden layers, sized according to the 'hidden_units' param.
    use_batch_norm = params['use_batch_norm']
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    net = fc.input_layer(features, params['feature_columns'])
    if use_batch_norm:
        net = tf.layers.batch_normalization(net, training=is_training)

    for units in params['hidden_units']:
        if use_batch_norm:
            x = tf.layers.dense(net, units=units, activation=None, use_bias=False)
            net = tf.nn.relu(tf.layers.batch_normalization(x, training=is_training))
        else:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    if use_batch_norm:
        x = tf.layers.dense(net, units=params['last_hidden_units'], activation=None, use_bias=False)
        net = tf.nn.elu(tf.layers.batch_normalization(x, training=is_training), name='user_vector_layer')
    else:
        net = tf.layers.dense(net, units=params['last_hidden_units'], activation=tf.nn.relu, name='user_vector_layer')

    return net


def _dnn_model_fn(features, labels, mode, params):

    layer = fc.input_layer(features, params['feature_columns'])
    layer = tf.layers.batch_normalization(layer, training=(mode == tf.estimator.ModeKeys.TRAIN))
    for units in params['hidden_units'][:-1]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
    layer = tf.layers.Dense(layer, params['hidden_units'][-1], activation=tf.nn.relu, name='user_vec_layer')

    cal_column = params['cal_column']
    if isinstance(cal_column, EmbeddingColumn):
        nce_weights = cal_column.get_dense_tensor()
    else:
        nce_weights = tf.Variable(tf.truncated_normal([params['n_classes'], params['last_hidden_units']],
                                                      stddev=1.0 / math.sqrt(params['last_hidden_units'])),
                                  name='nce_weights')

    nce_biases = tf.Variable(tf.zeros([params['n_classes']]), name='nce_biases')
    logits = tf.matmul(net, tf.transpose(nce_weights)) + nce_biases
    top_k_values, top_k_indices = tf.nn.top_k(logits, params["top_k"])

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=labels,
            inputs=layer,
            num_sampled=params['num_sampled'],
            num_classes=params['n_classes'],
            num_true=1,
            remove_accidental_hits=True,
            partition_strategy='div',
            name='match_model_nce_loss'))
        optimizer = optimizers.get_optimizer_instance(params["optimizer"], params["learning_rate"])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        estimator_spec = estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        precisions = params['eval_top_n'] if 'eval_top_n' in params else [5, 10, 20, 50, 100]
        metrics = {}
        for k in precisions:
            metrics["recall/recall@" + str(k)] = tf.metrics.recall_at_k(labels, logits, int(k))
            metrics["precision/precision@" + str(k)] = tf.metrics.precision_at_k(labels, logits, int(k))
            correct = tf.nn.in_top_k(logits, tf.squeeze(labels), int(k))
            metrics["accuary/accuary@" + str(k)] = tf.metrics.accuracy(labels=tf.ones_like(labels, dtype=tf.float32),
                                                                       predictions=tf.to_float(correct))
        labels_one_hot = tf.one_hot(labels, params['n_classes'])
        labels_one_hot = tf.reshape(labels_one_hot, (-1, params['n_classes']))
        print("labels_one_hot shape", labels_one_hot.get_shape())
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels_one_hot,
            logits=logits)
        loss = tf.reduce_mean(loss)
        estimator_spec = tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'user_vector': net,
            'top_k_values': top_k_values,
            'top_k_indices': top_k_indices,
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        estimator_spec = tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)
    return estimator_spec







class YTBDNNMatchEstimator:
    """
    an implement code after reading paper 'youtube recomendation with dnn'

    """

    def __init__(self, model_dir, params, config, warm_start_from=None):

        self._model_dir = model_dir
        self._params = params
        self._config = config
        self._warm_start_from = warm_start_from

    def parse_example(self, example):
        # 标记列
        label_column = fc.numeric_column("label", default_value=0, dtype=tf.int64)
        all_columns = [label_column]
        all_columns += self._params['feature_columns']
        example_spec = fc.make_parse_example_spec(all_columns)
        features = tf.parse_single_example(example, example_spec)
        labels = features.pop(label_column.name)
        return features, labels


    def init_estimator(self):
        classfier = tf.estimator.Estimator(model_fn=_dnn_model_fn,
                                           model_dir=self._model_dir,
                                           params=self._params,
                                           config=self._config,
                                           warm_start_from=self._warm_start_from)
        return classfier


    def train_and_evaluate(self, train_data, eval_data):

        classifier = self.init_estimator()



        batch_size = self._params['batch_size']
        shuffle_size = self._params['shuffle_size']
        train_steps = self._params['train_steps']
        #train
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: train_input_fn(train_data, batch_size, shuffle_size, self.parse_example),
            max_steps=train_steps)
        # eval
        input_fn_for_eval = lambda: eval_input_fn(eval_data, batch_size, self.parse_example)
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=600)
        #开始训练
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
        #评估结果
        results = classifier.evaluate(input_fn=input_fn_for_eval)
        for key in sorted(results): print('%s: %s' % (key, results[key]))
        print("after evaluate")
        return classifier





