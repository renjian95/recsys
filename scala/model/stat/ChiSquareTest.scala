package com.bj58.business.recsys.model.stat

import com.bj58.business.recsys.model.feature.LabeledPoint
import org.apache.spark.mllib.stat.test.ChiSqTest
import org.apache.spark.rdd.RDD
import org.apache.commons.math3.distribution.ChiSquaredDistribution
import org.apache.spark.ml.linalg.{DenseMatrix, Matrix}

import scala.collection.mutable

/**
  * @Author: renjian01
  * @Creation: 2019/4/1 09:44
  */


case class Method(name: String, chiSqFunc: (Double, Double) => Double)

object ChiSquareTest {


  private val maxCategories: Int = 10000

  // Pearson's chi-squared test
  val PEARSON = new Method("pearson", (observed: Double, expected: Double) => {
    val dev = observed - expected
    dev * dev / expected
  })

  /**
    * 特征-卡方检验-独立概率
    * @param data
    * @param method
    * @return
    */
  def chiSquaredFeatures(data: RDD[LabeledPoint], method: String): Array[ChiSquareTestResult] = {
    val numCols = data.first().features.size
    val results = new Array[ChiSquareTestResult](numCols)
    var labels: Map[Double, Int] = null

    val pairCounts = data.mapPartitions{ partition =>
      partition.flatMap{ case LabeledPoint(features, label) =>
        (0 until numCols).map{ idx =>
          val feature = features.apply(idx)
          (idx, feature, label)
        }
      }
    }.countByValue()

    if (labels == null) {
      labels = pairCounts.keys.filter(_._1 == 0).map(_._3).toArray.distinct.zipWithIndex.toMap
    }
    //label个数
    val numLabels = labels.size
    pairCounts.keys.groupBy(_._1).map { case (idx, kvs) =>
      val features = kvs.map(_._2).toArray.distinct.zipWithIndex.toMap
      val numFeatures = features.size
      val crossArr = new Array[Double](numFeatures * numLabels)
      kvs.foreach { case (_, feature, label) =>
        val i = features(feature)
        val j = labels(label)
        crossArr(numFeatures * j + i) += pairCounts((idx, feature, label))
      }
      val crossMatrix = new DenseMatrix(numFeatures, numLabels, crossArr)
      results(idx) = chiSquaredMatrix(crossMatrix, method)
    }
    results
  }




  def chiSquaredMatrix(counts: Matrix, methodName: String): ChiSquareTestResult = {
    val method = methodName match {
      case PEARSON.name => PEARSON
      case _ => throw new Exception("error")
    }
    val numRows = counts.numRows
    val numCols = counts.numCols

    // get row and column sums
    val colSums = new Array[Double](numCols)
    val rowSums = new Array[Double](numRows)
    val colMajorArr = counts.toArray
    val colMajorArrLen = colMajorArr.length

    var i = 0
    while (i < colMajorArrLen) {
      val elem = colMajorArr(i)
      if (elem < 0.0) {
        throw new IllegalArgumentException("Contingency table cannot contain negative entries.")
      }
      colSums(i / numRows) += elem
      rowSums(i % numRows) += elem
      i += 1
    }
    val total = colSums.sum

    // second pass to collect statistic
    var statistic = 0.0
    var j = 0
    while (j < colMajorArrLen) {
      val col = j / numRows
      val colSum = colSums(col)
      if (colSum == 0.0) {
        throw new IllegalArgumentException("Chi-squared statistic undefined for input matrix due to"
          + s"0 sum in column [$col].")
      }
      val row = j % numRows
      val rowSum = rowSums(row)
      if (rowSum == 0.0) {
        throw new IllegalArgumentException("Chi-squared statistic undefined for input matrix due to"
          + s"0 sum in row [$row].")
      }
      val expected = colSum * rowSum / total
      statistic += method.chiSqFunc(colMajorArr(j), expected)
      j += 1
    }
    val df = (numCols - 1) * (numRows - 1)
    if (df == 0) {
      // 1 column or 1 row. Constant distribution is independent of anything.
      // pValue = 1.0 and statistic = 0.0 in this case.
      new ChiSquareTestResult(1.0, 0, 0.0, methodName, "independent")
    } else {
      val pValue = 1.0 - new ChiSquaredDistribution(df).cumulativeProbability(statistic)
      new ChiSquareTestResult(pValue, df, statistic, methodName, "independent")
    }
  }

}



class ChiSquareTestResult (val pValue: Double, val degreesOfFreedom: Int,
                           val statistic: Double, val method: String,
                           val nullHypothesis: String) {

  override def toString: String = {

    // String explaining what the p-value indicates.
    val pValueExplain = if (pValue <= 0.01) {
      s"Very strong presumption against zero hypothesis: $nullHypothesis."
    } else if (0.01 < pValue && pValue <= 0.05) {
      s"Strong presumption against zero hypothesis: $nullHypothesis."
    } else if (0.05 < pValue && pValue <= 0.1) {
      s"Low presumption against zero hypothesis: $nullHypothesis."
    } else {
      s"No presumption against zero hypothesis: $nullHypothesis."
    }

    s"degrees of freedom = ${degreesOfFreedom.toString} \n" +
      s"statistic = $statistic \n" +
      s"pValue = $pValue \n" + pValueExplain
  }
}