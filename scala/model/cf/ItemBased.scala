package com.bj58.business.recsys.model.cf

import org.apache.spark.rdd.RDD

/**
  *
  *
  * @Author: renjian01
  * @Creation: 2018/12/23 16:04
  */

class ItemBased extends Serializable {

  //计算相似向量的个数
  private var numSimi = 100
  private var a = 0

  def setNumSimi(numSimi: Int): this.type = {
    this.numSimi = numSimi
    this
  }

  def train(df: RDD[(String,String,Double)]): Unit = {

  }

}
