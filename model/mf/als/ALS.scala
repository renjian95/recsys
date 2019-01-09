package com.bj58.business.recsys.model.mf.als

import org.apache.spark.{HashPartitioner, Partitioner}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable
import scala.collection.mutable.ArrayBuilder
import scala.reflect.ClassTag
/**
  *
  *
  * @Author: renjian01
  * @Creation: 2018/12/26 11:24
  */

//打分类
case class Rating(user: Int, item: Int, rating: Float)

//分块打分类
case class RatingBlock(userIds: Array[Int], itemIds: Array[Int], ratings: Array[Float]) {
  def size: Int = userIds.length
}

//生成RatingBlock
class RatingBlockBuilder extends Serializable {
  //评分矩阵中某个元素的用户id
  private val userIds = new mutable.ArrayBuilder.ofInt
  //评分矩阵中某个元素的物品id
  private val itemIds = new mutable.ArrayBuilder.ofInt
  //评分矩阵中某个元素的分值
  private val ratings = new mutable.ArrayBuilder.ofFloat
  //rating block中元素个数
  var size = 0

  /**
    *增加一个元素
    * @param element  Rating类型的评分矩阵元素
    */
  def add(element: Rating): this.type = {
    userIds += element.user
    itemIds += element.item
    ratings += element.rating
    size += 1
    this
  }

  /**
    * 与另一个rating block合并
    * @param ratingBlock
    * @return
    */
  def merge(ratingBlock: RatingBlock): this.type = {
    size += ratingBlock.size
    userIds ++= ratingBlock.userIds
    itemIds ++= ratingBlock.itemIds
    ratings ++= ratingBlock.ratings
    this
  }

  /**生成rating block
    * @return
    */
  def build() : RatingBlock = {
    RatingBlock(userIds.result(), itemIds.result(), ratings.result())
  }

}


class ALS extends Serializable {

  //分解后的用户，物品向量长度
  private var rank = 10

  def setRank(rank: Int): this.type = {
    this.rank = rank
    this
  }
  //迭代次数
  private var iterations = 10

  def setIterations(iterations: Int): this.type = {
    this.iterations = iterations
    this
  }
  //正则项系数
  private var lambda = 0.01
  def setLambda(lambda: Double): this.type = {
    this.lambda = lambda
    this
  }
  //模型选择(损失函数不同)
  private var method = "funksvd"
  //随机种子
  private var seed = 1234l
  /**设置随机种子
    * @param seed
    * @return
    */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  private var dataStorageLevel = StorageLevel.MEMORY_AND_DISK
  def setDataStorageLevel(level: StorageLevel): this.type = {
    this.dataStorageLevel = level
    this
  }
  //矩阵分块  按user向量或item向量分
  private var userBlocks = -1
  private var itemBlocks = -1

  def setBlocks(block: Int): this.type = {
    this.itemBlocks = block
    this.userBlocks = block
    this
  }

  def setUserBlocks(numUserBlocks: Int): this.type = {
    this.userBlocks = numUserBlocks
    this
  }

  def setItemBlocks(numItemBlocks: Int): this.type = {
    this.userBlocks = numItemBlocks
    this
  }


  /**
    * 将评分矩阵分块,假设评分矩阵数据分布均匀
    * @param ratings
    * @param userPartitioner
    * @param itemPartitioner
    */
  def partitionRating2Blocks(ratings: RDD[Rating],
                             userPartitioner: Partitioner,
                             itemPartitioner: Partitioner): RDD[((Int, Int), RatingBlock)] = {
    //评分矩阵总分块个数
    val numPartitions = userPartitioner.numPartitions * itemPartitioner.numPartitions
    //分块后数据
    ratings.mapPartitions { ratingIter =>
      val builders = Array.fill(numPartitions)(new RatingBlockBuilder)
      //分区生成((id1,id2),ratingblock)这样的数据
      ratingIter.flatMap { rating =>
        val userBlockId = userPartitioner.getPartition(rating.user)
        val itemBlockId = itemPartitioner.getPartition(rating.item)
        //矩阵按行分配id
        val id = userBlockId * itemPartitioner.numPartitions + itemBlockId
        val ratingBlockBuilder = builders(id)
        ratingBlockBuilder.add(rating)
        if (ratingBlockBuilder.size >= 2048) { // 2048 * (3 * 4) = 24k
          builders(id) = new RatingBlockBuilder
          Iterator.single(((userBlockId, itemBlockId), ratingBlockBuilder.build()))
        } else {
          Iterator.empty
        }
      } ++ {
        builders.zipWithIndex.filter(_._1.size > 0).map { case (block, idx) =>
          val userBlockId = idx / itemPartitioner.numPartitions
          val itemBlockId = idx % itemPartitioner.numPartitions
          ((userBlockId, userBlockId), block.build())
        }
      }
    }.groupByKey().mapValues { ratingBlocks =>
      val blockBuilder = new RatingBlockBuilder
      ratingBlocks.foreach(blockBuilder.merge)
      blockBuilder.build()
    }

  }

  def createDataBlocks(blockedRatings: RDD[((Int, Int), RatingBlock)],
                       srcPartitioner: Partitioner,
                       dstPartitioner: Partitioner): Unit = {

    val block1 = blockedRatings.map { case ((srcBlockId, dstBlockId), RatingBlock(srcIds, dstIds, ratings)) =>


    }

  }

  def train(ratings: RDD[Rating]): Unit = {
    val sc = ratings.sparkContext
    val numUserBlocks = if (this.userBlocks == -1) ratings.partitions.length / 2 else this.userBlocks
    val numItemBlocks = if (this.itemBlocks == -1) ratings.partitions.length / 2 else this.itemBlocks
    val userPart = new HashPartitioner(numUserBlocks)
    val itemPart = new HashPartitioner(numItemBlocks)

    //选择加速矩阵计算的方式
    val solver = new CholeskySolver
    //将评分矩阵数据分块
    val blockedRatings = partitionRating2Blocks(ratings, userPart, itemPart).persist(dataStorageLevel)

  }



}
