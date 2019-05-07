package com.bj58.business.recsys.model.tree.config

import com.bj58.business.recsys.model.tree.config.DTAlgo.{Classification, DTAlgo, Regression}
import com.bj58.business.recsys.model.tree.impurity.{Gini, Impurity, Variance}

/**
  * @Author: renjian01
  * @Creation: 2019/2/21 19:49
  */

class DTStrategy (var algo: DTAlgo,
                  var impurity: Impurity,
                  var numClasses: Int = 2,
                  var maxDepth: Int,
                  var maxBins: Int = 32,
                  var categoricalFeatures: Map[Int,Int],
                  var minInfoGain: Double = 0.0,
                  var subsamplingRate: Double = 1) {

  def this(algo: DTAlgo, impurity: Impurity,
           numClasses: Int, maxDepth: Int,
           maxBins: Int, categoricalFeatures: Map[Int,Int]) = {
    this(algo, impurity, numClasses, maxDepth, maxBins, categoricalFeatures, 0.0, 1.0)
  }


}

object DTStrategy {


  def defaultStrategy(algo: DTAlgo): DTStrategy = algo match {
    case DTAlgo.Classification =>
      new DTStrategy(Classification, Gini, 2, 10, 32, Map[Int,Int]())
    case DTAlgo.Regression =>
      new DTStrategy(Regression, Variance, 0, 10, 32, Map[Int,Int]())
  }
}
