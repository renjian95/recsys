package com.bj58.business.recsys.model.tree.impurity

/**
  * @Author: renjian01
  * @Creation: 2019/2/21 10:00
  */

trait Impurity extends Serializable {

  def calculate(labelCounts: Array[Double], totalCount: Double): Double

  def calcalate(sum: Double, squaredSum: Double, totalCount: Double): Double


}

object Gini extends Impurity {

  /**计算基尼不纯度  1 - \sum(p^2)
    *
    * @param labelCounts 各分类出现次数的集合
    * @param totalCount  总样本条数
    * @return  基尼不纯度
    */
  override def calculate(labelCounts: Array[Double], totalCount: Double): Double = {
    if (totalCount == 0) {
      return 0
    }
    val numClasses = labelCounts.length
    var index = 0
    var impurity = 1.0
    while (index < numClasses) {
      val p = labelCounts(index) / totalCount
      impurity -= p * p
      index += 1
    }
    impurity
  }

  override def calcalate(sum: Double, squaredSum: Double, totalCount: Double): Double =
    throw new UnsupportedOperationException("Gini.calculate")


}

object Variance extends Impurity {

  override def calcalate(sum: Double, squaredSum: Double, totalCount: Double): Double = {
    if (totalCount == 0) {
      return 0
    }
    (squaredSum - sum * sum / totalCount) / totalCount
  }

  override def calculate(counts: Array[Double], totalCount: Double): Double =
    throw new UnsupportedOperationException("Variance.calculate")

}
