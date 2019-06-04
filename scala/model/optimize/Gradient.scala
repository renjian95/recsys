package com.bj58.business.recsys.model.optimize

import com.bj58.business.recsys.model.feature.LabeledPoint
import com.bj58.business.recsys.model.util.math._
import breeze.linalg.{Vector, SparseVector, axpy}
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


class FMGradient(val ratio: Double) extends Gradient {

  override def compute(example: LabeledPoint, weights: Vector[Double], gradient: Vector[Double]): Double = {
    val features: SparseVector[Double] = example.features.asInstanceOf[SparseVector[Double]]
    val label = example.label
    val weightsArray = weights.toArray

    val k = ((weights.length - 1) / features.length) - 1
    val arr1 = (1 to k).toArray
    val arr2 = (0 until features.index.length)
    val linearValue = weightsArray(0) + arr2.map(i => weightsArray(features.index(i) * (k+1) + 1) * features.valueAt(i)).sum
    val mid = arr1.map(f => arr2.map(i => weightsArray(features.index(i) * (k+1) + 1 + f) * features.valueAt(i)).sum)
    val mid1 = mid.map(square).sum
    val mid2 = arr2.map(i => arr1.map(f => square(weightsArray(features.index(i) * (k+1) + 1 + f))).sum * square(features.valueAt(i))).sum
    val predict_y = linearValue + 0.5 * (mid1 - mid2)
    //println(s"linear: ${linearValue}, mid1: ${mid1}, mid2: ${mid2},predict_y: ${predict_y}")
    val partOfGradient = sigmoid(predict_y) - label
    val exampleWeight = if (example.label > 0) ratio else 1.0

    //计算梯度部分
    val index: Array[Int] = Array.fill[Int](features.index.length * (k+1) + 1)(0)
    val value: Array[Double] = Array.fill[Double](features.index.length * (k+1) + 1)(1.0)
    for (i <- arr2) {
      val idx = features.index(i)
      (0 to k).foreach { f =>
        val j = idx * (k+1) + 1 + f
        val v = features.valueAt(i)
        index(i * (k+1) + 1 + f) = j
        value(i * (k+1) + 1 + f) = if (f == 0) v else v * mid(f-1) + square(v) * weightsArray(j)
      }
    }
    axpy(partOfGradient * exampleWeight, (new SparseVector[Double](index, value, weightsArray.length)).asInstanceOf[Vector[Double]], gradient)
    //返回损失
    if (label > 0) log1pExp(-predict_y) * exampleWeight
    else (log1pExp(-predict_y) + predict_y) * exampleWeight
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
