package com.bj58.business.recsys.model.classification.fm

import breeze.linalg.{DenseVector, SparseVector, Vector}
import com.bj58.business.recsys.model.feature.LabeledPoint
import com.bj58.business.recsys.model.optimize.{ElasticNetUpdater, FMGradient, GradientDescent, LogisticGradient}
import com.bj58.business.recsys.model.util.XORShiftRandom
import org.apache.spark.rdd.RDD

/**
  * @Author: renjian01
  * @Creation: 2019-05-29 21:23
  */

class FM extends Serializable {

  //优化算法
  private var algo = "sgd"
  def setAlgo(algo: String): this.type = {
    this.algo = algo
    this
  }
  //特征隐向量长度
  private var implicitK = 3
  def setImplicitK(k: Int): this.type = {
    this.implicitK = k
    this
  }
  //迭代次数
  private var iter = 10
  def setIter(iter: Int): this.type = {
    this.iter = iter
    this
  }
  //学习率(步长)
  private var learningRate = 0.1
  def setLearningRate(learningRate: Double): this.type = {
    this.learningRate = learningRate
    this
  }

  //正则项系数
  private var regParam = 0.0
  def setRegParam(reg: Double) : this.type = {
    this.regParam = reg
    this
  }

  //分配 l1和l2正则
  private var elasticNet = 0.0
  def setElasticNetParam(elasticNet: Double): this.type = {
    this.elasticNet = elasticNet
    this
  }
  //随机种子
  private var seed = 1234l
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  //抽样比例
  private var fraction = 1.0
  def setFraction(fraction: Double): this.type = {
    this.fraction = fraction
    this
  }

  //正负样本比例
  private var ratio = 10.0
  def setRatio(ratio: Double): this.type = {
    this.ratio = ratio
    this
  }

  def train(input: RDD[LabeledPoint]): FMModel = {
    //特征数
    val featureNum = input.first().features.length
    algo match {
      case "sgd" =>
        //设置随机数生成器 借用spark原生随机数生成器 速度更快
        val initRandom = new XORShiftRandom(seed)
        val initWeightArray = Array.fill[Double](1 + featureNum * (implicitK + 1))(initRandom.nextGaussian() / 10)
        initWeightArray(0) = 0.0
        (0 until featureNum).foreach(x => initWeightArray(1 + x * (implicitK + 1)) = 0.0)
        //初始化  initRandom.nextFloat()生成[0,1]中随机数v
        val initWeights = new DenseVector[Double](initWeightArray)
        //val initWeights = Vector.zeros[Double](1 + featureNum * (implicitK + 1)).toDenseVector
        val gd = new GradientDescent(new FMGradient(ratio), new ElasticNetUpdater).
          setBatchFraction(fraction).
          setNumIterations(iter).
          setStepSize(learningRate).
          setRegParam(Array(regParam * elasticNet, regParam * (1 - elasticNet)))
        val (weights, loss) = gd.optimize(input, initWeights)
        FMModel(weights, implicitK).setLoss(loss)
      case _ =>
        System.out.println("sorry, not support")
        throw new UnsupportedOperationException
    }

  }


  def trainFromModel(input: RDD[LabeledPoint], model: FMModel): Unit = {
    //特征数
    val featureNum = input.first().features.length
    algo match {
      case "sgd" =>
        val initWeights = model.getWeights
        //val initWeights = Vector.zeros[Double](1 + featureNum * (implicitK + 1)).toDenseVector
        val gd = new GradientDescent(new FMGradient(ratio), new ElasticNetUpdater).
          setBatchFraction(fraction).
          setNumIterations(iter).
          setStepSize(learningRate).
          setRegParam(Array(regParam * elasticNet, regParam * (1 - elasticNet)))
        val (weights, loss) = gd.optimize(input, initWeights)
        model.update(weights).setLoss(loss)
      case _ =>
        System.out.println("sorry, not support")
        throw new UnsupportedOperationException

    }

  }




}







