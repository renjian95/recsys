#!/usr/bin/env python
# encoding: utf-8
'''
@author: renjian
@contact: 1838915959@qq.com
@file: train.py
@time: 2019-05-02 23:50
@desc:
'''

import os
import json
import random
from model.ytb_dnn_match import ytb_dnn_match
from tensorflow import feature_column as fc
import tensorflow as tf



flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model_output", "Path to the training data.")
flags.DEFINE_string("output_item_vector", "./item_vector_output/nce_weights.ckpt", "Path to the trained item vector.")
flags.DEFINE_string("train_data", "data/samples", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "data/eval", "Path to the evaluation data.")
flags.DEFINE_integer("n_classes", 150000, "The number of possible classes/labels")
flags.DEFINE_integer("num_sampled", 500, "The number of negative classes to randomly sample per batch.")
flags.DEFINE_string("hidden_units", "128", "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("last_hidden_units", "64", "last hidden layer of the NN, equal to user vector")
flags.DEFINE_string("eval_top_n", "20,50,100,200,300,500",
                    "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("train_steps", 200000, "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
flags.DEFINE_integer("top_k", 20, "predict the top k results")
flags.DEFINE_integer("shuffle_size", 10000, "dataset shuffle buffer size")
flags.DEFINE_float("learning_rate", 0.005, "Learning rate")
flags.DEFINE_float("dropout_rate", 0.0, "Drop out rate")
flags.DEFINE_integer("num_parallel_readers", 5, "number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "Save checkpoints every this many steps")
flags.DEFINE_string("optimizer", "Adagrad", "the name of optimizer")
flags.DEFINE_string("ps_hosts", "s-xiasha-10-2-176-43.hx:2222", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "s-xiasha-10-2-176-42.hx:2223,s-xiasha-10-2-176-44.hx:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_boolean("run_on_cluster", False, "Whether the cluster info need to be passed in as input")
flags.DEFINE_boolean("use_batch_norm", False, "Whether to use batch normalization for hidden layers")
flags.DEFINE_boolean("predict", False, "Whether to predict")
FLAGS = flags.FLAGS


def set_tfconfig_environ():
    if "TF_CLUSTER_DEF" in os.environ:
        cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
        task_index = int(os.environ["TF_INDEX"])
        task_type = os.environ["TF_ROLE"]

        tf_config = dict()
        worker_num = len(cluster["worker"])
        if task_type == "ps":
            tf_config["task"] = {"index": task_index, "type": task_type}
            FLAGS.job_name = "ps"
            FLAGS.task_index = task_index
        else:
            if task_index == 0:
                tf_config["task"] = {"index": 0, "type": "chief"}
            else:
                tf_config["task"] = {"index": task_index - 1, "type": task_type}
            FLAGS.job_name = "worker"
            FLAGS.task_index = task_index

        if worker_num == 1:
            cluster["chief"] = cluster["worker"]
            del cluster["worker"]
        else:
            cluster["chief"] = [cluster["worker"][0]]
            del cluster["worker"][0]

        tf_config["cluster"] = cluster
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        print("TF_CONFIG", json.loads(os.environ["TF_CONFIG"]))

    if "INPUT_FILE_LIST" in os.environ:
        INPUT_PATH = json.loads(os.environ["INPUT_FILE_LIST"])
        if INPUT_PATH:
            print("input path:", INPUT_PATH)
            FLAGS.train_data = INPUT_PATH.get(FLAGS.train_data)
            FLAGS.eval_data = INPUT_PATH.get(FLAGS.eval_data)
        else:  # for ps
            print("load input path failed.")
            FLAGS.train_data = None
            FLAGS.eval_data = None



def parse_argument():
  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index == "":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)
  os.environ["TF_ROLE"] = FLAGS.job_name
  os.environ["TF_INDEX"] = str(FLAGS.task_index)

  # Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")
  cluster = {"worker": worker_spec, "ps": ps_spec}
  os.environ["TF_CLUSTER_DEF"] = json.dumps(cluster)




def build_feature_columns():
    #点击历史
    clicks = fc.categorical_column_with_hash_bucket("click", 1000000, dtype=tf.int64)
    clicks_weighted = fc.weighted_categorical_column(clicks, 'click_weights')
    clicks_embed = fc.embedding_column(clicks_weighted, 64)
    #city
    sloc1 = fc.categorical_column_with_hash_bucket("sloc1", 1000, dtype=tf.int32)
    sloc1_embed = fc.embedding_column(sloc1, 16)
    #特征列
    feature_columns = [clicks_embed, sloc1_embed]
    return feature_columns, clicks_embed




def main(unused_argv):
    #设置运行环境
    set_tfconfig_environ()

    feature_columns, cal_column = build_feature_columns()
    model_dir = FLAGS.model_dir
    params = {
        'feature_columns': feature_columns,
        'cal_column': cal_column,
        'hidden_units': FLAGS.hidden_units.split(','),
        'last_hidden_units': FLAGS.last_hidden_units,
        'optimizer': FLAGS.optimizer,
        'learning_rate': FLAGS.learning_rate,
        'dropout_rate': FLAGS.dropout_rate,
        'n_classes': FLAGS.n_classes,
        'num_sampled': FLAGS.num_sampled,
        'batch_size': FLAGS.batch_size,
        'shuffle_size': FLAGS.shuffle_size,
        'train_steps': FLAGS.train_steps,
        'top_k': FLAGS.top_k,
        'eval_top_n': FLAGS.eval_top_n.split(',')
    }
    config = tf.estimator.RunConfig(save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    # 生成模型
    estimate = ytb_dnn_match.YTBDNNMatchEstimator(model_dir, params, config)
    classifier = estimate.train_and_evaluate(FLAGS.train_data, FLAGS.eval_data)

    if FLAGS.predict:
        pred_data = FLAGS.pred_data
        if isinstance(pred_data, str) and os.path.isdir(pred_data):
            eval_files = [pred_data + '/' + x for x in os.listdir(pred_data)]
        else:
            eval_files = pred_data
        input_fn_for_pred = lambda: estimate.eval_input_fn(eval_files, params['batch_size'])
        pred = list(classifier.predict(input_fn=input_fn_for_pred))
        # print("pred result example", next(pred))
        random.shuffle(pred)
        print("pred result example", pred[:50])

    # elif FLAGS.job_name == "worker" and FLAGS.task_index == 0:
    #     print("exporting model ...")
    #     feature_spec = tf.feature_column.make_parse_example_spec(my_feature_columns)
    #     print(feature_spec)
    #     serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    #     classifier.export_savedmodel(FLAGS.output_model, serving_input_receiver_fn)
    #
    #     print("save item vector")
    #     nce_weights = classifier.get_variable_value('nce_weights')
    #     nce_biases = classifier.get_variable_value('nce_biases')
    #     [rows, cols] = nce_weights.shape
    #     with tf.gfile.FastGFile(FLAGS.output_item_vector, 'w') as f:
    #         for i in range(rows):
    #             f.write(unicode(str(i) + "\t"))
    #             for j in range(cols):
    #                 f.write(unicode(str(nce_weights[i, j])))
    #                 f.write(u',')
    #             f.write(unicode(str(nce_biases[i])))
    #             f.write(u'\n')
    # print("quit main")



if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    if FLAGS.run_on_cluster: parse_argument()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)









