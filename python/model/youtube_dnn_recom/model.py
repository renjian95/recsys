#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:50:02 2019

@author: renjian
"""
import tensorflow as tf
import time
import pickle
import random
import numpy as np


class Dnn:
    """
    Dnn类实现了论文：deep nerual networks for youtube recommendations
    中的matching阶段模型
    """
    
    def __init__(self, args):
        self.item_embedding_size = args.item_embedding_size
        self.sloc_embedding_size = args.sloc_embedding_size
        self.item_count = args.item_count
        self.sloc_count = args.sloc_count
        self.checkpoint_dir = args.checkpoint_dir
        self.learning_rate = args.learning_rate
        
        
    def train(self, sample):
        #解析sample样本
        click_sequences = sample[0]
        sub_example = sample[1]
        sub_label = sample[2]
        sloc = sample[4]
        hist_size = sample[5]
        #构建输入特征 build feature
        item_embedding_weights = tf.get_variable(name='item_embedding_weights',
                                                 shape=[self.item_count, self.item_embedding_size])
        item_embedding_bias = tf.get_variable(name='item_embedding_bias',
                                              shape=[self.item_count], 
                                              initializer=tf.constant_initializer(0.0))
        sloc_embedding_weights = tf.get_variable(name='sloc_embedding_weights',
                                                 shape=[self.sloc_count, self.sloc_embedding_size])
        
        click_embedding = tf.nn.embedding_lookup(item_embedding_weights, click_sequences)
        
#        mask = tf.sequence_mask(self.hist_size, tf.shape(average_click_embedding)[1], dtype=tf.float32)  # [B,T]
#        mask = tf.expand_dims(mask, -1)  # [B,T,1]
#        mask = tf.tile(mask, [1, 1, tf.shape(average_click_embedding)[2]])  # [B,T,3*e]
#        click_embedding *= mask  # [B,T,3*e]
        
        average_click_embedding = tf.reduce_sum(click_embedding, 1)  #[*,128]
        div_embedding = tf.tile(tf.expand_dims(hist_size, 1), [1, self.item_embedding_size]) #除数
        average_click_embedding = tf.div(average_click_embedding, 
                                         tf.cast(div_embedding, tf.float32))
        
        geo_embedding = tf.nn.embedding_lookup(sloc_embedding_weights, sloc) 
        
        embedding_feature = tf.concat([average_click_embedding, 
                                       geo_embedding], axis=-1)
    
        #构建网络 build net
        input_layer = tf.layers.batch_normalization(inputs=embedding_feature, 
                                                    name='input_layer')
        dense_layer1 = tf.layers.dense(inputs=input_layer, 
                                       units=1024,
                                       activation=tf.nn.relu, 
                                       name='dense_layer1')
        dense_layer2 = tf.layers.dense(inputs=dense_layer1, 
                                       units=512,
                                       activation=tf.nn.relu, 
                                       name='dense_layer2')
        dense_layer3 = tf.layers.dense(inputs=dense_layer2, 
                                       units=256,
                                       activation=tf.nn.relu, 
                                       name='dense_layer3')
        #输出用户向量层
        dense_layer4 = tf.layers.dense(inputs=dense_layer3, 
                                       units=self.item_embedding_size,
                                       activation=tf.nn.relu, 
                                       name='dense_layer4')
        #softmax层
        user_embedding = tf.expand_dims(dense_layer4, 1)
                #softmax层需要迭代的参数
        example_embedding_weights = tf.nn.embedding_lookup(item_embedding_weights, sub_example)
        example_embedding_weights = tf.transpose(example_embedding_weights, perm=[0, 2, 1])
        example_embedding_bias = tf.nn.embedding_lookup(item_embedding_bias, sub_example)
                #损失函数 优化  迭代
        logits = tf.squeeze(tf.matmul(user_embedding, example_embedding_weights), axis=1) + example_embedding_bias
        predict_label = tf.nn.softmax(logits)
        origin_label = tf.cast(sub_label, dtype=tf.float32)
        self.loss = tf.reduce_mean(-origin_label * tf.log(predict_label))
#        self.loss = tf.reduce_mean(
#                tf.nn.softmax_cross_entropy_with_logits(logits=logits,
#                                                        labels=origin_label)                                                                       )
#                )
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params), 
                                                       self.global_step)
#        self.train_op = self.optimizer.minimize(loss=self.loss, 
#                                                global_step=global_step)        
    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)
          
            
            
            
            
         
#解析tfrecords文件
def example_parser(serialized_example):
    context_features = {
        "uid": tf.FixedLenFeature([], dtype=tf.int64),
        "sloc": tf.FixedLenFeature([], dtype=tf.int64),
        "hist_size": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "hist": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "sub_example": tf.FixedLenSequenceFeature([], dtype=tf.int64),
        "sub_label": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    uid = context_parsed['uid']
    sloc = context_parsed['sloc']
    hist_size = context_parsed['hist_size']
    click_sequences = sequence_parsed['hist']
    sub_example = sequence_parsed['sub_example']
    sub_label = sequence_parsed['sub_label']
    return click_sequences, sub_example, sub_label, uid, sloc, hist_size


def gen_batch_input(tfrecords_file, example_parser, batch_size, num_epochs, padded_shapes):
    dataset = tf.contrib.data.TFRecordDataset(tfrecords_file) \
        .map(example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes) \
        .repeat(num_epochs)
    return dataset.make_one_shot_iterator().get_next()


class Args():
    item_embedding_size = 128
    sloc_embedding_size = 5
    item_count = -1
    sloc_count = -1
    learning_rate = 1.0
    

if __name__ == '__main__':
    
    base_path = '/home/hdp_lbg_ectech/renjian01/recsys/bash/esc'
    checkpoint_path = base_path+'/checkpoint'
    
    args = Args()  #初始化参数
    with open(base_path+'/args_data.pkl', 'rb') as f:
        ent_count, sloc_count, click_data_count = pickle.load(f)
        
    args.checkpoint_dir = base_path+'/save_path/ckpt'
    args.item_count = ent_count
    args.sloc_count = sloc_count
    #生成批数据
    inputs = gen_batch_input(base_path+'/train.tfrecords',
                             example_parser, 
                             64, 3, 
                             ([None], [None], [None], [], [], [])
                             )
    #启动计算
    with tf.Session() as sess:
        
        model = Dnn(args)
        model.train(inputs)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 开启一个协调器
        coord = tf.train.Coordinator()
        # 使用start_queue_runners 启动队列填充
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sum_loss = 0.0
        start_time = time.time()
        try:
            while not coord.should_stop():
                loss, _ = sess.run([model.loss, model.train_op])
                sum_loss += loss
                if model.global_step.eval() % 1000 == 0:
                    model.save(sess, checkpoint_path)
                    average_loss = sum_loss / model.global_step.eval()
                    print('当前迭代%d次, 平均误差为:%.5f' % (model.global_step.eval(), average_loss))
        except tf.errors.OutOfRangeError:
            train_time = (time.time() - start_time) / 60
            print('结束训练, 耗时: %.2f分钟' % train_time)
            model.save(sess, checkpoint_path)
        finally:
            coord.request_stop()
            
        coord.join(threads)
