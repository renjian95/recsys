#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:49:58 2019

@author: renjian
"""

import pandas as pd
import random
import pickle
import numpy as np
import tensorflow as tf
import time
import math
import bisect

df = pd.read_csv('/Users/renjian/Desktop/portrait_eval.csv', header=0)
df1 = df[['类型', '价格', '品牌', '车系']]

# 读取数据
base_path = '/home/hdp_lbg_ectech/renjian01/recsys/bash/esc'
user_click_hist_data_path = base_path+'/user_click_history_data.txt'
user_click_hist_data = pd.read_csv(user_click_hist_data_path, sep='\t', header=None).head(10000)
user_click_hist_data.columns = ['uid', 'infoid', 'sloc1', 'timestamp']
user_click_hist_data = user_click_hist_data.drop_duplicates()

#定义取用户最近点击序列长度
sequence = 20
int_max = 2147483647
negative = 15
random.seed(1234)

# 构建 映射hash 函数
def build_map(df, col_name):
    key = df[col_name].unique().tolist()
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key

#开始构建
user_map, uid = build_map(user_click_hist_data, 'uid')
ent_map, entid = build_map(user_click_hist_data, 'infoid')
sloc_map, sloc = build_map(user_click_hist_data, 'sloc1')
ent_count, sloc_count, click_data_count = len(entid), len(sloc), len(user_click_hist_data)

#负采样概率表函数
def create_cum_table(power, ent_count, ent_grouped_count):
    table_size = int_max
    cum_table = np.zeros(ent_count, np.int64)
    pow_sum = ent_grouped_count.map(lambda x: math.pow(x, power)).sum()
    cumulative = 0.0
    index = 0
    while index < ent_count:
        cumulative += math.pow(ent_grouped_count[index], power)
        cum_table[index] = round(cumulative / pow_sum * table_size)
        index += 1
    if len(cum_table) > 0: 
        cum_table[len(cum_table) - 1] = table_size
    return cum_table
#生成概率表
ent_grouped_count = user_click_hist_data['infoid'].groupby(user_click_hist_data['infoid']).count()
cum_table = create_cum_table(0.75, ent_count, ent_grouped_count)
    

def get_train_records(uid, sloc, sub_sample, label, hist, hist_size):
    frame_hist = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), hist))
    frame_sub_sample = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), sub_sample))
    frame_label = list(map(lambda id: tf.train.Feature(int64_list=tf.train.Int64List(value=[id])), label))
    
    example = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=[uid])),
                'sloc': tf.train.Feature(int64_list=tf.train.Int64List(value=[sloc])),
                'hist_size': tf.train.Feature(int64_list=tf.train.Int64List(value=[hist_size]))
            }),
        feature_lists=tf.train.FeatureLists(feature_list={
            'hist': tf.train.FeatureList(feature=frame_hist),
            'sub_example': tf.train.FeatureList(feature=frame_sub_sample),
            'sub_label': tf.train.FeatureList(feature=frame_label)
        })
    )
    return example.SerializeToString()
    
    
def generate_neg(ent_list, cum_table):
    neg = ent_list[0]
    while neg in ent_list:
        rd = random.randint(0, int_max)
        neg = bisect.bisect_left(cum_table, rd)
    return neg
    
    
def generate_tfrecord(user_click_hist_data, 
                      ent_count, 
                      cum_table, 
                      tfrecord_file):
    #输出文件
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    #构建采样
    for uid, hist in user_click_hist_data.groupby('uid'):
        ent_list = hist['infoid'].tolist()
        if len(ent_list) < 2:
            continue
        sloc_list = hist['sloc1'].tolist()
        #生成负样本
        neg_list = [generate_neg(ent_list, cum_table) for i in range(negative * len(ent_list))]
        neg_list = np.array(neg_list)
        
        for i in range(1, len(ent_list)):
            hist = ent_list[:i]
            hist_size = len(hist)
            cur_click = ent_list[i]
            cur_sloc = sloc_list[i]
            neg_index = np.random.randint(len(neg_list), size=negative)
            neg_click = list(neg_list[neg_index])
            sub_sample = [cur_click] + neg_click
            label = np.zeros(len(sub_sample), np.int64)
            label[0] = 1
            train_records = get_train_records(uid, cur_sloc, sub_sample, label, hist, hist_size)
            writer.write(train_records)
            
    writer.close()

     
    
args_path = base_path+'/args_data.pkl'
with open(args_path, 'wb') as f:
    pickle.dump((ent_count, sloc_count, click_data_count), f, pickle.HIGHEST_PROTOCOL)
    
tfrecord_file = base_path+'/train.tfrecords'
generate_tfrecord(user_click_hist_data, ent_count, cum_table, tfrecord_file)
