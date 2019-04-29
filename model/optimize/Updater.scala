package com.bj58.business.recsys.model.optimize

import scala.math._
import breeze.linalg.{Vector, axpy, norm}
import com.bj58.business.recsys.model.feature.LabeledPoint
import com.bj58.business.recsys.model.optimize.LBFGS.State
import org.apache.spark.rdd.RDD
import com.bj58.business.recsys.model.util.math._

/**
  * @Author: renjian01
  * @Creation: 2019/4/7 10:53
  *
  * 用于和Gradient类配合，完成基于梯度的优化
  *
  * Gradient类用于计算梯度
  * Updater类用于更新
  */

trait Updater extends Serializable {

  /**
    * 用于更新参数向量 并 返回新的正则项
    *
    * @param lastWeights - 原参数向量
    * @param gradient - 梯度向量
    * @param stepSize - 步长
    * @param iter - 迭代次数
    * @param regParam - 正则项参数
    *
    * @return (newWeights, regularization)
    */
  def compute(lastWeights: Vector[Double],
              gradient: Vector[Double],
              stepSize: Double,
              iter: Int,
              regParam: Array[Double]): (Vector[Double], Double)
}

trait NewtonUpdater extends Serializable {

  //计算损失和梯度
  def calLossGrad(weights: Vector[Double]): (Double, Vector[Double])

  def compute(state: State): Vector[Double]
}


/**
  * 无正则项，且步长随迭代衰减
  */
class BaseUpdater extends Updater {

  override def compute(lastWeights: Vector[Double],
                       gradient: Vector[Double],
                       stepSize: Double,
                       iter: Int,
                       regParam: Array[Double]): (Vector[Double], Double) = {
    val newStepSize = stepSize / math.sqrt(iter)
    val newWeights: Vector[Double] = lastWeights.toDenseVector
    axpy(-newStepSize, gradient, newWeights)
    (newWeights, 0)
  }
}

/**
    带有了l1正则项
  */
class L1Updater extends Updater {
  override def compute(lastWeights: Vector[Double],
                       gradient: Vector[Double],
                       stepSize: Double,
                       iter: Int,
                       regParam: Array[Double]): (Vector[Double], Double) = {
    //先更新由损失函数确认的部分梯度
    val newStepSize = stepSize / math.sqrt(iter)
    val newWeights: Vector[Double] = lastWeights.toDenseVector
    axpy(-newStepSize, gradient, newWeights)
    //更新由l1正则项带来的梯度
    val lambda1 = if (regParam.length > 0) regParam(0) else 0.0
    val regStepSize = lambda1 * newStepSize  //正则项的学习率
    axpy(-regStepSize, lastWeights.map(signum), newWeights)
    //返回新w向量 与 新正则项值
    (newWeights, norm(newWeights, 1.0) * lambda1)
  }
}

/**
  * 带有了l2正则项
  * 0.5 x ||w||^2
  */
class L2Updater extends Updater {
  override def compute(lastWeights: Vector[Double],
                       gradient: Vector[Double],
                       stepSize: Double,
                       iter: Int,
                       regParam: Array[Double]): (Vector[Double], Double) = {
    val newStepSize = stepSize / math.sqrt(iter) //学习率
    val lambda2 = if (regParam.length > 1) regParam(1) else 0.0
    val regStepSize = lambda2 * newStepSize  //正则项的学习率
    val newWeights: Vector[Double] = lastWeights.toDenseVector
    newWeights :*= (1.0 - regStepSize)   //更新正则项梯度
    axpy(-newStepSize, gradient, newWeights) //更新损失函数梯度
    val norm2: Double = norm(newWeights, 2.0)  //l2
    (newWeights, 0.5 * lambda2 * norm2 * norm2)
  }
}


/**
  * 弹性网络正则项
  */
class ElasticNetUpdater extends Updater {

  override def compute(lastWeights: Vector[Double],
                       gradient: Vector[Double],
                       stepSize: Double,
                       iter: Int,
                       regParam: Array[Double]): (Vector[Double], Double) = {
    val newStepSize = stepSize / math.sqrt(iter) //学习率
    val lambda1 = if (regParam.length > 0) regParam(0) else 0.0
    val lambda2 = if (regParam.length > 1) regParam(1) else 0.0
    val l1StepSize = lambda1 * newStepSize  //l1正则项的学习率
    val l2StepSize = lambda2 * newStepSize  //l2正则项的学习率
    val newWeights: Vector[Double] = lastWeights.toDenseVector
    newWeights :*= (1.0 - l2StepSize)   //更新l2正则项梯度
    axpy(-newStepSize, gradient, newWeights) //更新损失函数梯度
    axpy(-l1StepSize, lastWeights.map(signum), newWeights)  //更新l1正则项梯度
    val norm2 = norm(newWeights, 2.0)  //l2
    val norm1 = norm(newWeights, 1.0)
    (newWeights, lambda1 * norm1 + 0.5 * lambda2 * norm2 * norm2)
  }
}


class ElasticNetNewtonUpdater(data: RDD[LabeledPoint], numExamples: Long,
                              gradient: Gradient, regParam: Array[Double]) extends NewtonUpdater {

  override def calLossGrad(weights: Vector[Double]): (Double, Vector[Double]) = {
    val weightSize = weights.size
    //将参数向量广播
    val bcWeights = data.sparkContext.broadcast(weights)
    //累积梯度、损失
    val (cumGradient, cumLoss) = data.treeAggregate((Vector.zeros[Double](weightSize), 0.0))(
      seqOp = (u, example) => {
        val l = gradient.compute(example, bcWeights.value, u._1)
        (u._1, u._2 + l)
      },
      combOp = (u, v) => {
        (u._1 += v._1, u._2 + v._2)
      })
    bcWeights.destroy()
    //正则项参数
    val lambda1 = if (regParam.length > 0) regParam(0) else 0.0
    val lambda2 = if (regParam.length > 1) regParam(1) else 0.0
    val norm2 = norm(weights, 2.0)  //l2
    val norm1 = norm(weights, 1.0)  //l1
    val regularization = lambda1 * norm1 + 0.5 * lambda2 * norm2 * norm2
    //计算损失与正则项修正后的损失
    val loss = cumLoss / numExamples
    val adjustedLoss = loss + regularization
    //计算梯度
    val grad = cumGradient / numExamples.toDouble
    axpy(lambda2, weights, grad)
    axpy(lambda1, weights.map(signum), grad)
    (adjustedLoss, grad)
  }

  override def compute(state: State): Vector[Double] = {
    //下降方向
    val direction = state.hessian.mulGrad(state.grad)
    //最优步长
    val stepSize = 1.0 //待修改
    val newWeights: Vector[Double] = state.weights.toDenseVector
    axpy(stepSize, direction, newWeights)
    newWeights
  }
}


