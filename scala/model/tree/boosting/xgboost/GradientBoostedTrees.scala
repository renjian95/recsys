package com.bj58.business.recsys.model.tree.boosting.xgboost

import com.bj58.business.recsys.model.tree.config.BoostingStrategy
import com.bj58.business.recsys.model.tree.loss.Loss
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}
import org.apache.spark.ml.tree.impl.RandomForest
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.model.DecisionTreeModel

/**
  * @Author: renjian01
  * @Creation: 2019-04-22 13:39
  */

class GradientBoostedTrees (val boostingStrategy: BoostingStrategy) extends Serializable {

  def train(data: RDD[LabeledPoint]): (Array[DecisionTreeModel], Array[Double]) = {
    val algo = boostingStrategy.treeStrategy.algo
    algo match {
      case Algo.Regression =>
        GradientBoostedTrees.boost(data, boostingStrategy)
      case Algo.Classification =>
        // Map labels to -1, +1 so binary classification can be treated as regression.
        val tranData = data.map(x => new LabeledPoint((x.label * 2) - 1, x.features))
        GradientBoostedTrees.boost(tranData, boostingStrategy)
      case _ =>
        throw new IllegalArgumentException(s"$algo is not supported by gradient boosting.")
    }
  }

}


object GradientBoostedTrees {


  def predict(originPrediction: Double, tree: DecisionTreeModel,
              weight: Double, features: Vector): Double = {
    originPrediction + weight * tree.predict(features)
  }


  def calInitPredictionError(data: RDD[LabeledPoint],
                         initTreeWeight: Double,
                         initTree: DecisionTreeModel,
                         loss: Loss):  RDD[(Double, Double)] = {
    data.map { lp =>
      data.first().features
      val pred = predict(0.0, initTree, initTreeWeight, lp.features)
      val error = loss.computeError(pred, lp.label)
      (pred, error)
    }
  }

  def updatePredictionError(data: RDD[LabeledPoint], predictionAndError: RDD[(Double, Double)],
                             treeWeight: Double, tree: DecisionTreeModel,
                             loss: Loss): RDD[(Double, Double)] = {

    val newPredError = data.zip(predictionAndError).mapPartitions { iter =>
      iter.map { case (lp, (pred, error)) =>
        val newPred = predict(pred, tree, treeWeight, lp.features)
        val newError = loss.computeError(newPred, lp.label)
        (newPred, newError)
      }
    }
    newPredError
  }

  def boost(input: RDD[LabeledPoint], boostingStrategy: BoostingStrategy): (Array[DecisionTreeModel], Array[Double]) = {
    //初始化决策树模型及权重容器
    val numIterations = boostingStrategy.numIterations
    val modelArr = new Array[DecisionTreeModel](numIterations)
    val weightArr = new Array[Double](numIterations)
    //初始化损失计算
    val loss = boostingStrategy.loss
    val learningRate = boostingStrategy.learningRate
    //再次确认boost使用的基模型的参数，防止出错
    val treeStrategy = boostingStrategy.treeStrategy.copy
    treeStrategy.algo = Algo.Regression
    treeStrategy.impurity = Variance
    //确认数据cached
    if (input.getStorageLevel == StorageLevel.NONE) input.persist(StorageLevel.MEMORY_AND_DISK)

    //初始化第一颗树
    val firstTreeModel = new DecisionTree(treeStrategy).run(input)
    val firstTreeWeight = 1.0
    modelArr(0) = firstTreeModel
    weightArr(0) = firstTreeWeight

    //第一棵树的预测值与误差
    var predError: RDD[(Double, Double)] = calInitPredictionError(input, firstTreeWeight, firstTreeModel, loss)

    var m = 1
    var doneLearning = false
    while (m < numIterations && !doneLearning) {
      // 拟合残差
      val data = predError.zip(input).map { case ((pred, _), point) =>
        LabeledPoint(-loss.gradient(pred, point.label), point.features)
      }
      //迭代模型
      val model = new DecisionTree(treeStrategy).run(data)
      modelArr(m) = model
      weightArr(m) = learningRate
      //更新预测值与误差
      predError = updatePredictionError(input, predError, weightArr(m), modelArr(m), loss)
      m += 1
    }

    input.unpersist()

    (modelArr, weightArr)
  }
}
