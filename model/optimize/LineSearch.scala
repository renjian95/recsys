package com.bj58.business.recsys.model.optimize


import breeze.linalg.Vector
/**
  * @Author: renjian01
  * @Creation: 2019-04-21 21:33
  */

trait LineSearch {

  def minimize(init: Double = 1.0):Double
}



class StrongWolfeLineSearch extends LineSearch {

  override def minimize(init: Double): Double = {

    0.0
  }
}


object LineSearch {


}
