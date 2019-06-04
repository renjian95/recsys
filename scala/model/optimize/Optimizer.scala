package com.bj58.business.recsys.model.optimize



import com.bj58.business.recsys.model.feature.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.Vector
/**
  * @Author: renjian01
  * @Creation: 2019/3/14 10:17
  */

trait Optimizer extends Serializable {

  def optimize(data: RDD[LabeledPoint], initialWeights: Vector[Double]): (Vector[Double], Double)

}
