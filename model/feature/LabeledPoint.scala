package com.bj58.business.recsys.model.feature


import breeze.linalg.Vector

/**
  *
  *
  * @Author: renjian01
  * @Creation: 2019/1/7 21:12
  */

case class LabeledPoint(features: Vector[Double], label: Double) {
  override def toString: String = {
    s"($label,$features)"
  }
}
