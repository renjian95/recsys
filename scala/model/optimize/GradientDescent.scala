package com.bj58.business.recsys.model.optimize

import com.bj58.business.recsys.model.feature.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.{Vector, norm}

import scala.collection.mutable

/**
  * @Author: renjian01
  * @Creation: 2019/3/26 11:24
  */

class GradientDescent(private var gradient: Gradient,
                      private var updater: Updater) extends Optimizer {

  private var batchFraction = 1.0
  private var numIterations = 10
  private var stepSize = 0.1
  private var regParam = Array(0.0, 0.0)
  private var minFraction = 0.000001


  /**
    * 设置抽样比例
    * @param fraction 比例
    * @return
    */
  def setBatchFraction(fraction: Double): this.type = {
    this.batchFraction = fraction
    this
  }


  /**
    * 设置SGD的迭代次数.
    */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
    * 设置步长
    * @param step
    * @return
    */
  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  /**
    * 设置正则项参数
    */
  def setRegParam(regParam: Array[Double]): this.type = {
    this.regParam = regParam
    this
  }

  /**
    * 设置误差下限
    * @param minFraction
    * @return
    */
  def setMinFraction(minFraction: Double): this.type = {
    this.minFraction = minFraction
    this
  }


  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  override def optimize(data: RDD[LabeledPoint], initialWeights: Vector[Double]): (Vector[Double], Double) = {

    val (weights, loss) = GradientDescent.run(data, initialWeights, gradient, updater,
      batchFraction, numIterations, stepSize, regParam, minFraction)
    (weights, loss)
  }

}


object GradientDescent {

  def Converged(lastWeights: Vector[Double], currentWeights: Vector[Double], minFraction: Double): Boolean = {
    //两轮迭代参数之差的范数
    val diff: Double = norm(lastWeights - currentWeights, 2)
    diff < minFraction * Math.max(norm(currentWeights), 1.0)
  }

  def run(data: RDD[LabeledPoint], initialWeights: Vector[Double],
          gradient: Gradient, updater: Updater,
          batchFraction: Double, numIterations: Int,
          stepSize: Double, regParam: Array[Double],
          minFraction: Double): (Vector[Double], Double) = {
    //记录每轮迭代损失值
    val lossArr = mutable.ArrayBuilder.make[Double]
    val numExamples = data.count()
    if (numExamples == 0) {
      System.out.println("未找到训练数据")
      return (initialWeights, -1.0)
    }
    require(numExamples * batchFraction > 1, "每轮迭代样本最少为1")

    var lastWeights: Vector[Double] = null
    var currentWeights: Vector[Double] = null
    //参数向量
    var weights = initialWeights.copy
    var finalLoss: Double = -1.0
    val weightSize = weights.length
    //初始化正则项
    var regularization = updater.compute(weights, Vector.zeros(weightSize), 0, 1, regParam)._2
    var idx = 1
    var isConverged = false
    while (idx <= numIterations && !isConverged) {
      //将参数向量广播
      val bcWeights = data.sparkContext.broadcast(weights)
      //累积梯度、损失、样本量
      val (cumGradient, cumLoss, batchSize) = data.sample(false, batchFraction, idx).
        treeAggregate((Vector.zeros[Double](weightSize), 0.0, 0L))(
          seqOp = (u, example) => {
            val l = gradient.compute(example, bcWeights.value, u._1)
            (u._1, u._2 + l, u._3 + 1)
          },
          combOp = (u, v) => {
            (u._1 += v._1, u._2 + v._2, u._3 + v._3)
          })
      val lastLoss = if (lossArr.result().length > 0) lossArr.result().toSeq.reverse.head else Int.MaxValue
      val loss = cumLoss / batchSize + regularization
      lossArr += loss
      val adjust = if (loss - lastLoss < 0 && idx > 1) idx / 2 else idx

      val update = updater.compute(weights, cumGradient / batchSize.toDouble, stepSize, adjust, regParam)
      weights = update._1
      regularization = update._2
      //判断迭代是否收敛
      lastWeights = currentWeights
      currentWeights = weights
      if (lastWeights != null && currentWeights != null) {
        isConverged = Converged(lastWeights, currentWeights, minFraction)
      }
      System.out.println(s"第${idx}次迭代损失为：${loss}")
      //System.out.println(s"第${idx}次迭代梯度为：${cumGradient / batchSize.toDouble}")
      finalLoss = loss
      //bcWeights.destroy()  广播变量销毁可能出错 spark bug
      idx += 1
    }
    (weights, finalLoss)
  }
}