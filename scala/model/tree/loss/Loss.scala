package com.bj58.business.recsys.model.tree.loss

/**
  * @Author: renjian01
  * @Creation: 2019-04-22 14:15
  */

trait Loss extends Serializable {


  def gradient(prediction: Double, label: Double): Double

  def computeError(prediction: Double, label: Double): Double
}
