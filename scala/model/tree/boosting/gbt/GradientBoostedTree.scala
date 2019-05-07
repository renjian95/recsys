package com.bj58.business.recsys.model.tree.boosting.gbt

import com.bj58.business.recsys.model.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.rdd.RDD

/**
  *
  *
  * @Author: renjian01
  * @Creation: 2019/1/7 20:48
  */

class GradientBoostedTree(val algo: String) extends Serializable {

  //迭代次数 即树个数
  private var numIterations = 3
  //树的深度
  private var maxDepth = 5
  //分类树时采用  默认为2   当前只支持2分类
  private var numClasses = 2
  //列抽样
  private var featureSubsetStrategy = "all"
  //随机种子
  private var seed = 0l

  def setNumIterations(numIterations: Int): this.type = {
    this.numIterations = numIterations
    this
  }

  def setMaxDepth(maxDepth: Int): this.type = {
    this.maxDepth = maxDepth
    this
  }

  def setNumClasses(numClasses: Int): this.type = {
    this.numClasses = numClasses
    this
  }

  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  def setFeatureSubsetStrategy(featureSubsetStrategy: String): this.type = {
    this.featureSubsetStrategy = featureSubsetStrategy
    this
  }


  def updatePrediction(
                        features: Vector,
                        prediction: Double,
                        tree: DecisionTreeRegressionModel,
                        weight: Double): Double = {
    prediction
  }

  def train(data: RDD[LabeledPoint]): Unit = {
    val input = algo match {
      case s: String if "regression".equals(s) || "Regression".equals(s) => data
      case _ => data.map(x => LabeledPoint(x.features, x.label * 2 - 1))
    }
    boost(input, false)
  }

  def boost(input: RDD[LabeledPoint], validate: Boolean): Unit = {
    //初始化
    val treeModels = new Array[DecisionTreeRegressionModel](numIterations)
    val treeModelWeights = new Array[Double](numIterations)

  }

}
