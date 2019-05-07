package com.bj58.business.recsys.model.tree.config


import com.bj58.business.recsys.model.tree.loss.Loss
import org.apache.spark.mllib.tree.configuration.Strategy

import scala.beans.BeanProperty

/**
  * @Author: renjian01
  * @Creation: 2019-04-22 13:27
  */

case class BoostingStrategy (@BeanProperty var treeStrategy: Strategy,
                             @BeanProperty var loss: Loss,
                             @BeanProperty var numIterations: Int = 50,
                             @BeanProperty var learningRate: Double) extends Serializable

object BoostingStrategy {



}
