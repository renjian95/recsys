#!/usr/bin/env python
# encoding: utf-8
'''
@author: renjian
@contact: 1838915959@qq.com
@file: esmm.py
@time: 2019-03-06 18:19
@desc:
'''


import tensorflow as tf
from tensorflow import estimator
from tensorflow import metrics
from tensorflow.python.estimator.canned import optimizers
from tensorflow import nn
import math
import os
from tensorflow import feature_column as fc
from model.utils.tfrecords_op import *



def build_base_net(features, params, mode):
    net = fc.input_layer(features, params['feature_columns'])
    net = tf.layers.batch_normalization(net, training=(mode == tf.estimator.ModeKeys.TRAIN))
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            net = tf.layers.dropout(net, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
    #logits
    logits = tf.layers.dense(net, 1, activation=None)
    return logits


def dnn_model_fn(features, labels, mode, params):
    with tf.variable_scope('ctr'):
        ctr_logits = build_base_net(features, params, mode)

    with tf.variable_scope('cvr'):
        cvr_logits = build_base_net(features, params, mode)

    ctr_pred = tf.sigmoid(ctr_logits, name='ctr_pred')
    cvr_pred = tf.sigmoid(ctr_logits, name='cvr_pred')
    ctcvr_pred = tf.multiply(ctr_pred, cvr_pred, name='ctcvr_pred')

    ctr_label = labels['ctr']
    cvr_label = labels['cvr']
    ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_label, logits=ctr_logits),
                             name="ctr_loss")
    ctcvr_loss = tf.reduce_mean(tf.losses.log_loss(labels=cvr_label, predictions=ctcvr_pred))
    loss = tf.add(ctr_loss, ctcvr_loss, name='loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        estimator_spec = estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        ctr_accuracy = metrics.accuracy(labels=ctr_label, predictions=tf.to_float(tf.greater_equal(ctr_pred, 0.5)))
        ctcvr_accuracy = metrics.accuracy(labels=cvr_label, predictions=tf.to_float(tf.greater_equal(ctcvr_pred, 0.5)))
        ctr_auc = metrics.auc(labels=ctr_label, predictions=ctr_pred)
        ctcvr_auc = metrics.auc(labels=cvr_label, predictions=ctcvr_pred)
        metric_results = {'ctcvr_auc': ctcvr_auc,
                          'ctcvr_accuracy': ctcvr_accuracy,
                          'ctr_accuracy': ctr_accuracy,
                          'ctr_auc': ctr_auc}
        estimator_spec = estimator.EstimatorSpec(mode, loss, eval_metric_ops=metric_results)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'ctcvr_p': ctcvr_pred,
            'ctr_p': ctr_pred,
            'cvr_p': cvr_pred
        }
        export_outputs = {'prediction': estimator.export.PredictOutput(predictions)}
        estimator_spec = estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    return estimator_spec



class ESMM:

    def __init__(self, model_dir, params, config, warm_start_from=None):

        self._model_dir = model_dir
        self._params = params
        self._config = config
        self._warm_start_from = warm_start_from


    def parse_example(self, example):
        click_label = fc.numeric_column('click_label', default_value=0, dtype=tf.int64)
        conversion_label = fc.numeric_column('conversion_label', default_value=0, dtype=tf.int64)
        all_columns = [click_label, conversion_label]
        all_columns += self._params['feature_columns']
        example_spec = fc.make_parse_example_spec(all_columns)
        features = tf.parse_single_example(example, example_spec)
        click_label = features.pop(click_label.name)
        conversion_label = features.pop(conversion_label.name)
        labels = {'ctr': tf.to_float(click_label), 'cvr': tf.to_float(conversion_label)}
        return features, labels



    def init_estimator(self):
        estimate = tf.estimator.Estimator(model_fn=dnn_model_fn,
                                           model_dir=self._model_dir,
                                           params=self._params,
                                           config=self._config,
                                           warm_start_from=self._warm_start_from)
        return estimate

    def train_and_evaluate(self, train_data, eval_data):
        estimate = self.init_estimator()

        batch_size = self._params['batch_size']
        shuffle_size = self._params['shuffle_size']
        train_steps = self._params['train_steps']
        # train
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: train_input_fn(train_data, batch_size, shuffle_size, self.parse_example),
            max_steps=train_steps)
        # eval
        input_fn_for_eval = lambda: eval_input_fn(eval_data, batch_size, self.parse_example)
        eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_for_eval, throttle_secs=600)
        # 开始训练
        tf.estimator.train_and_evaluate(estimate, train_spec, eval_spec)
        # 评估结果
        results = estimate.evaluate(input_fn=input_fn_for_eval)
        for key in sorted(results): print('%s: %s' % (key, results[key]))
        print("after evaluate")
        return estimate
