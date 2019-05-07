package com.bj58.business.recsys.model.tree.config

/**
  * @Author: renjian01
  * @Creation: 2019/2/21 19:47
  */

object DTAlgo extends Enumeration {

  type DTAlgo = Value

  val Classification, Regression = Value

  def matchString(str: String): DTAlgo = {
    str match {
      case "classfication" | "Classfication" => Classification
      case "regression" | "Regression" => Regression
      case _ => throw new IllegalArgumentException("illegal algorithm choice")
    }
  }
}
