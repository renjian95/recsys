package com.bj58.business.recsys.model.classification.fm

import breeze.linalg
import com.bj58.business.recsys.model.util.math._
import breeze.linalg.{DenseVector, SparseVector, Vector}
import com.bj58.business.recsys.model.feature.Measure
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import breeze.numerics._
import org.apache.spark.SparkContext
import org.apache.spark.sql.{Row, SaveMode, SparkSession}

import scala.collection.mutable
/**
  * @Author: renjian01
  * @Creation: 2019-05-30 13:04
  */
case class Data(w0: Double, k: Int, matrix: Seq[Seq[Double]])

class FMModel(private var w0: Double,
              private var k: Int,
              private var matrix: Array[Array[Double]]) extends Serializable {


  def getW0: Double = w0


  def getK: Int = k

  def getMatrix: Array[Array[Double]] = matrix


  def getWeights: Vector[Double] = new DenseVector[Double](Array.concat(Array(w0), matrix.flatMap(x => x)))
  /**
    * 非one-hot下的预测   没写完 有时间再写
    * @param xn
    */
  def predict(xn: SparseVector[Double]): Unit = {
    val index = xn.index
    val value = xn.data
    val linearValue = w0
  }

  private var measure: Measure = Measure()
  def getMeasure: Measure = measure

  def setLoss(loss: Double): this.type = {
    this.measure.loss = loss
    this
  }


  def predictOnLine(index: Array[Int]): Double = {
    //特征所在的参数
    val midMatrix = index.map(i => matrix(i))
    val linearValue = midMatrix.map(_(0)).sum + w0
    val mid1 = (1 to k).map(f => square(midMatrix.map(_(f)).sum)).sum
    val mid2 = midMatrix.flatMap(x => x.slice(1, k+1).map(square)).sum
    val predict_y = linearValue + 0.5 * (mid1 - mid2)
    //println(s"linear: ${linearValue}, mid1: ${mid1}, mid2: ${mid2}, predict_y： ${predict_y}")
    sigmoid(predict_y)
  }

  def measureOnLine(test: RDD[(Array[Int], Double)]): Measure = {
    val scoreAndLabel = test.map(x => (predictOnLine(x._1), x._2))
    val binaryMetrics = new BinaryClassificationMetrics(scoreAndLabel, 100)
    val auc = binaryMetrics.areaUnderROC()
    this.measure.auc = auc
    measure
  }


  def update(weights: Vector[Double]): this.type = {
    val weightArray = weights.toArray
    val w0 = weightArray(0)

    val featureNum = (weightArray.length - 1) / (k + 1)
    val matrix = Array.ofDim[Double](featureNum, k + 1)

    for (i <- 0 until featureNum) {
      for (j <- 0 until k + 1) {
        matrix(i)(j) = weightArray(i * (k + 1) + j + 1)
      }
    }
    this.w0 = w0
    this.matrix = matrix
    this
  }

  def save(spark: SparkSession, path: String): Unit = {

    val data = Data(w0, k, matrix.map(_.toSeq).toSeq)
    spark.createDataFrame(Seq(data)).repartition(1).
      write.mode(SaveMode.Overwrite).parquet(path)
  }



}


object FMModel {



  def apply(weights: Vector[Double], implicitK: Int): FMModel = {

    val weightArray = weights.toArray
    val w0 = weightArray(0)

    val featureNum = (weightArray.length - 1) / (implicitK + 1)
    val matrix = Array.ofDim[Double](featureNum, implicitK + 1)

    for (i <- 0 until featureNum) {
      for (j <- 0 until implicitK + 1) {
        matrix(i)(j) = weightArray(i * (implicitK + 1) + j + 1)
      }
    }

    new FMModel(w0, implicitK, matrix)
  }


  def load(spark: SparkSession, path: String): FMModel = {
    val df = spark.read.parquet(path)
    val dataArray = df.select("w0", "k", "matrix").take(1)
    val data = dataArray(0)
    val (w0, k, matrix) = data match {
      case Row(w0: Double, k: Int, matrix: Seq[Seq[Double]]) =>
        (w0, k, matrix)
    }
    new FMModel(w0, k, matrix.map(_.toArray).toArray)
  }

}
