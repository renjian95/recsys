//package com.bj58.business.recsys.model.tree
//
//import com.bj58.business.recsys.model.tree.config.DTStrategy
//import com.bj58.business.recsys.model.tree.config.DTAlgo.Regression
//import com.bj58.business.recsys.model.util.XORShiftRandom
//import org.apache.spark.ml.feature.LabeledPoint
//import org.apache.spark.rdd.RDD
//
//import scala.collection.mutable
//import scala.tools.nsc.util.HashSet
//
///**
//  * @Author: renjian01
//  * @Creation: 2019/2/21 13:17
//  */
//
//class RandomForest(val strategy: DTStrategy,
//                   val numTrees: Int,
//                   val featureSubset: String,
//                   val seed: Long) extends Serializable {
//
//
//  def this(strategy: DTStrategy, numTrees: Int, featureSubset: String) =
//    this(strategy, numTrees, featureSubset, 1234l)
//
//
//  def findSplits4ContinuousFeature(idx: Int, values: Iterable[Double],
//                                   maxBins: Array[Int]): Array[Double] = {
//    val splits: Array[Double] = if (values.isEmpty) {
//      Array.empty[Double]
//    } else {
//      val numSplits = maxBins(idx)
//      val numValues = values.size
//      val valueCountMap = values.groupBy(x => x).mapValues(_.size)
//      val valueCount = valueCountMap.toSeq.sortBy(_._1).toArray
//      val possilbeSplits = valueCount.length - 1
//      if (possilbeSplits == 0) {
//        Array.empty[Double] //特征值为常量
//      } else if (possilbeSplits <= numSplits) {
//        (0 until possilbeSplits)
//          .map(x => (valueCount(x)._1 + valueCount(x+1)._1) / 2.0)
//          .toArray
//      } else {
//        val stride = numValues.toDouble / (numSplits + 1)  //分箱步长
//        val splitBuild = mutable.ArrayBuilder.make[Double]()
//        var index = 1
//        var currentCount = valueCount(0)._2
//        var targetCount = stride
//        while (index < valueCount.length) {
//          val previousCount = currentCount
//          currentCount += valueCount(index)._2
//          val previousGap = math.abs(previousCount - targetCount)
//          val currentGap = math.abs(currentCount - targetCount)
//          if (previousGap < currentGap) {
//            splitBuild += (valueCount(index - 1)._1 + valueCount(index)._1) / 2.0
//            targetCount += stride
//          }
//          index += 1
//        }
//        splitBuild.result()
//      }
//    }
//    splits
//  }
//
//
//  def findSplits(input: RDD[LabeledPoint], continuousFeatures: IndexedSeq[Int],
//                 maxBins: Array[Int]): Array[Array[String]] = {
//
//    val continuousSplits = input
//      .flatMap(point => continuousFeatures.map(idx => (idx, point.features(idx))))
//      .groupByKey(math.min(continuousFeatures.length, input.partitions.length))
//      .map{ case (idx, values) =>
//        val p = findSplits4ContinuousFeature(idx, values, maxBins)
//        (idx, p)
//      }.collectAsMap()
//
//
//
//
//  }
//
//  def run(input: RDD[LabeledPoint]): RandomForestModel = {
//
//    //构建准备数据
//    val numFeatures = input.first().features.size  //特征数
//    val numExamples = input.count()  //样本数
//    val numClasses = strategy.numClasses    //几分类
//    val maxBins = strategy.maxBins  //划分
//    if (strategy.categoricalFeatures.nonEmpty) {
//      val maxCategories = strategy.categoricalFeatures.values.max
//      val maxCategoryFeature = strategy.categoricalFeatures.find(_._2==maxCategories).get._1
//      require(maxBins >= maxCategories, "maxCategories > maxBins")
//    }
//    //特征分箱数 及 无序特征集合
//    val numBins = Array.fill[Int](numFeatures)(maxBins - 1)
//    val unOrderedFeatures = new mutable.HashSet[Int]()
//    if (numClasses <= 2) {
//      strategy.categoricalFeatures.foreach { case (index, num) =>
//        if (num > 1) {
//          numBins(index) = num - 1
//        }
//      }
//    } else { //多分类
//
//      strategy.categoricalFeatures.foreach { case (index, num) =>
//        if (num > 1) {
//          if (true) { //无序特征
//            unOrderedFeatures.add(index)
//            numBins(index) = (1 << (num - 1)) - 1
//          } else {
//            numBins(index) = num - 1
//          }
//
//        }
//      }
//    }
//    //每棵树采用多少特征
//    val featureSubsetStrategy = featureSubset match {
//      case "auto" =>
//        if (numTrees == 1) {
//          "all"
//        } else {
//          if (strategy.algo == Regression) "onethird" else "sqrt"
//        }
//      case _ => featureSubset
//    }
//    val numFeatures4Tree = featureSubsetStrategy match {
//      case "all" => numFeatures
//      case "sqrt" => math.sqrt(numFeatures).ceil.toInt
//      case "onethird" => (numFeatures / 3.0).ceil.toInt
//      case z if z.matches("[0-9]+") => math.min(z.toInt, numFeatures)
//      case d if d.matches("0\\.[0-9]+") => (d.toDouble * numFeatures).ceil.toInt
//      case _ => numFeatures
//    }
//
//    val continuousFeatures = (0 until numFeatures).
//      filter(x => !strategy.categoricalFeatures.contains(x))
//    val sample = if (continuousFeatures.nonEmpty) {
//      val numSample = math.max(maxBins * maxBins, 100000)
//      val fraction = if (numSample < numExamples) numSample / numExamples.toDouble else 1.0
//      input.sample(false, fraction, new XORShiftRandom(seed).nextInt())
//    } else {
//      input.sparkContext.emptyRDD[LabeledPoint]
//    }
//
//
//
//    new RandomForestModel(Array())
//  }
//
//
//}
//
//object RandomForest extends Serializable {
//
//  def train(input: RDD[LabeledPoint],
//            strategy: DTStrategy,
//            numTrees: Int,
//            featureSubsetStrategy: String): RandomForestModel = {
//    val rf = new RandomForest(strategy, numTrees, featureSubsetStrategy)
//    rf.run(input)
//  }
//
//
//}
