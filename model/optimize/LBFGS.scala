package com.bj58.business.recsys.model.optimize


import com.bj58.business.recsys.model.feature.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.Vector
import breeze.optimize.{CachedDiffFunction, DiffFunction, FirstOrderException, FirstOrderMinimizer, NaNHistory, LBFGS => BreezeLBFGS}
import breeze.linalg._
import com.bj58.business.recsys.model.optimize.GradientDescent.Converged

/*
  * @Author: renjian01
  * @Creation: 2019/3/14 10:15
  */

class LBFGS (private var gradient: Gradient,
             private var updater: NewtonUpdater) extends Optimizer {


  //最大迭代次数
  private var numIterations = 10
  //最小误差，两轮迭代之间误差低于这个值  停止迭代
  private var minFraction = 1E-4
  //正则项的系数
  private var regParam = Array(0.0, 0.0)
  //内存 一般设为3-7
  private var numSearchMemory = 10

  def setMinFraction(error: Double): this.type = {
    this.minFraction = minFraction
    this
  }

  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  def setRegParam(regParam: Array[Double]): this.type = {
    this.regParam = regParam
    this
  }

  def setNumSearchMemory(numSearchMemory: Int): this.type = {
    this.numSearchMemory = numSearchMemory
    this
  }



  override def optimize(data: RDD[LabeledPoint], initialWeights: Vector[Double]): Vector[Double] = {

    val weights = LBFGS.run(gradient, updater, initialWeights,
      numSearchMemory, minFraction, numIterations)
    weights
  }
}


object LBFGS {

  case class State(weights: Vector[Double],
                   loss: Double, grad: Vector[Double],
                   iter: Int,
                   initLoss: Double,
                   hessian: HessianMatrix)

  //黑赛矩阵类  p表示 x(k+1)-x(k)   q表示 grad(k+1)-grad(k)
  case class HessianMatrix(val lm: Int = 10,
                           val weightDelta: IndexedSeq[Vector[Double]] = IndexedSeq.empty,
                           val gradDelta: IndexedSeq[Vector[Double]] = IndexedSeq.empty) {

    //累积次数
    val length = weightDelta.length
    /**
      * 计算黑赛矩阵乘梯度向量
      * @param grad 梯度向量
      * @return 下降方向向量
      */
    def mulGrad(grad: Vector[Double] ): Vector[Double] = {
      //初始化对角阵h0
      val initDiag = if (length == 0) {
        1.0
      } else{
        val p = weightDelta.head
        val q = gradDelta.head
        val ptq: Double = p.dot(q)
        val qtq: Double = q.dot(q)
        math.abs(ptq / qtq)
      }

      //通过两层循环计算下降方向，用来代替 -inv(Hessian)*grad
      val direction = grad.copy
      val ganma = new Array[Double](lm)
      val rho = new Array[Double](lm)
      for(i <- 0 until length) {
        rho(i) = weightDelta(i).dot(gradDelta(i))
        ganma(i) = weightDelta(i).dot(direction) / rho(i)
        if(ganma(i).isNaN) {
          throw new NaNHistory
        }
        axpy(-ganma(i), gradDelta(i), direction)
      }
      direction *= initDiag
      for(i <- (length - 1) to 0 by (-1)) {
        val beta = gradDelta(i).dot(direction) / rho(i)
        axpy(ganma(i) - beta, weightDelta(i), direction)
      }

      direction *= -1.0
      direction
    }

    //更新黑赛矩阵，返回新矩阵
    def update(newWeightDelta: Vector[Double] , newGradDelta: Vector[Double]): LBFGS.HessianMatrix = {
      val a = (newWeightDelta +: weightDelta).take(lm)
      val b = (newGradDelta +: gradDelta).take(lm)
      new HessianMatrix(lm, a, b)
    }

  }


  /**
    * 判断是否收敛
    * @param minFraction
    * @return
    */
  def Converged(minFraction: Double): Boolean = {
    true  //待完成
  }


  def run(gradient: Gradient, updater: NewtonUpdater,
          initialWeights: Vector[Double], numSearchMemory: Int,
          minFraction: Double, numIterations: Int): Vector[Double] = {

    //参数向量
    var weights = initialWeights.copy
    val weightSize = weights.length
    //初始化状
    val (initLoss, initGrad) = updater.calLossGrad(weights)
    var state = State(weights, initLoss, initGrad, 0, initLoss, new HessianMatrix(numSearchMemory))

    var idx = 1
    var isConverged = false //是否收敛
    while (idx <= numIterations && !isConverged) {
      //更新参数、损失、梯度
      weights = updater.compute(state)
      val (loss, grad) = updater.calLossGrad(weights)
      //相对上一次迭代的改善
      val improvement = (state.loss - loss) / (state.loss.abs max loss.abs max 1E-6 * state.initLoss.abs)
      //更新黑赛矩阵近似
      val hessian = state.hessian.update(weights - state.weights, grad - state.grad)
      state = State(weights, loss, grad, idx, state.initLoss, hessian)
      //收敛判断
      isConverged = Converged(minFraction)
      idx += 1
    }

    weights
  }
}
