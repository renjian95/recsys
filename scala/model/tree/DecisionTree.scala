//package com.bj58.business.recsys.model.tree
//
//
//import com.bj58.business.recsys.model.tree.config.{DTAlgo, DTStrategy}
//import org.apache.spark.ml.feature.LabeledPoint
//import org.apache.spark.rdd.RDD
//
//
//
///**
//  * @Author: renjian01
//  * @Creation: 2019/2/20 21:07
//  */
//
//class DecisionTree(val strategy: DTStrategy, val seed: Long)
//  extends Serializable {
//
//  def this(strategy: DTStrategy) = this(strategy, 1234l)
//
//  def run(input: RDD[LabeledPoint]): DecisionTreeModel = {
//    val randomForest = new RandomForest(strategy, 1, "all", seed)
//    val randomForestModel = randomForest.run(input)
//    randomForestModel.trees(0)
//  }
//
//}
//
//
//object DecisionTree extends Serializable {
//
//  def train(input: RDD[LabeledPoint], strategy: DTStrategy): DecisionTreeModel = new DecisionTree(strategy).run(input)
//
//
//}
//
//
//
//
//
//
