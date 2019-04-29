package com.bj58.business.recsys.model.optimize

import com.bj58.business.recsys.model.feature.LabeledPoint
import com.bj58.business.recsys.model.util.math._
import breeze.linalg.{Vector, axpy}
import breeze.numerics._

/**
  * @Author: renjian01
  * @Creation: 2019/3/14 11:04
  */

trait Gradient extends Serializable {

  /**
    * 更新梯度   计算损失函数值
    * @param example 样本
    * @param weights 参数向量
    * @param gradient 累积梯度
    * @return
    */
  def compute(example: LabeledPoint, weights: Vector[Double], gradient: Vector[Double]): Double


  /**
    * 计算梯度与损失
    * @param example
    * @param weights
    * @return
    */
  def compute(example: LabeledPoint, weights: Vector[Double]): (Vector[Double], Double) = {
    val gradient = Vector.zeros[Double](weights.size)
    val loss = compute(example, weights, gradient)
    (gradient, loss)
  }
}



/**
  * 最小二乘方法 损失函数梯度
  */
class LeastSquaresGradient extends Gradient {
  override def compute(example: LabeledPoint, weights: Vector[Double]): (Vector[Double], Double) = {
    val features = example.features
    val label = example.label
    val diff = features.dot(weights) - label
    val loss = diff * diff / 2.0
    val gradient = features.copy
    scal(diff, gradient)
    (gradient, loss)
  }

  override def compute(example: LabeledPoint, weights: Vector[Double], gradient: Vector[Double]): Double = {
    val features = example.features
    val label = example.label
    val diff = features.dot(weights) - label
    val loss = diff * diff / 2.0
    axpy(diff, features, gradient)
    loss
  }
}


/** 逻辑回归的损失函数梯度
  *  xi * (yi - sigmod(wx))
  * @param numClasses
  */
class LogisticGradient(var numClasses: Int) extends Gradient {

  def this() = this(2)

  /**设置逻辑回归分类数
    * @param numClasses
    * @return
    */
  def setNumClasses(numClasses: Int): this.type = {
    this.numClasses = numClasses
    this
  }

  /**
    * 更新梯度   计算损失函数值
    * @param example 样本
    * @param weights 参数向量
    * @param gradient
    * @return   损失
    */
  override def compute(example: LabeledPoint, weights: Vector[Double], gradient: Vector[Double]): Double = {
    val features = example.features
    val label = example.label
    val dataSize = features.size
    val weightSzie = weights.size
    require(weightSzie % dataSize == 0 && weightSzie / dataSize + 1 == numClasses, "参数个数不对")

    numClasses match {
      case 2 =>
        val linearValue: Double = weights dot features
        val partOfGradient = sigmoid(linearValue) - label
        axpy(partOfGradient, features, gradient)
        if (label > 0) log1pExp(-linearValue) else log1pExp(-linearValue) + linearValue
      case x: Int if x > 2 =>
        0.0
    }
  }
}



/**
  * SVM分类使用的损失函数
  *
  * hinge loss -- 合页损失函数
  */
class HingeGradient extends Gradient {

  override def compute(example: LabeledPoint, weights: Vector[Double], gradient: Vector[Double]): Double = {
    0.0
  }

}
