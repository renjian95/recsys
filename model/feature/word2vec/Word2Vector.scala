package com.bj58.business.recsys.model.feature.word2vec

import com.alibaba.fastjson.JSON
import com.bj58.business.recsys.model.util.XORShiftRandom
import com.bj58.business.recsys.util.ArrayLikeHelper
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.RangePartitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.collection.mutable
/**
  *认为spark ml封装word2vector存在问题且不完善，重写word2vec
  * 修复了并行化中参数更新问题，拓展CBOW模型  拓展H-softmax训练方法(原模型只包含用 负采样优化的skip-gram)
  *
  * @Author: renjian01
  * @Creation: 2018/12/14 18:02
  */

//语料库中 词对象 H-softmax使用
case class CorpusWord(var word: String,  //词
                      var cn: Int,       //出现次数
                      var point: Array[Int], //从根节点到叶子节点的路径
                      var code: Array[Int], //词在huffman tree中的二进制编码
                      var codeLen: Int //编码长度
                      )


class Word2Vector extends Serializable {

  //词向量的维度
  private var vectorSize = 100
  //优化算法的学习率
  private var learningRate = 0.025
  //训练方法，skip-gram:1 或 CROB:0
  private var algorithm = 1
  //优化方法  hierarchical softmax:1 或 negative-sample:0
  private var hsns = 1
  //如果采用 negative-sample 优化方法 负样本个数usually between 5-20
  private var negative = 5
  //表示将训练样本切分成多少个batch
  private var numPartitions = 1
  //迭代次数，通常输入一个batch的数据，迭代一次
  private var numIterations = 1
  //随机种子
  private var seed = 1234l
  //词出现最低次数
  private var minCount = 5
  //句子最长长度
  private var maxSentenceLength = 1000
  //窗口大小
  private var window = 5
  //文本总词汇长度
  private var corpusSize = 0
  //文本中出现词总数
  private var corpusCount = 0L
  //词汇表数组
  @transient private var corpus: Array[CorpusWord] = null
  //词库的hash表
  @transient private var corpusHash = mutable.HashMap.empty[String, Int]
  //负采样时概率分布表
  @transient private var cumTable: Array[Int] = null

  /**
    * 使用H-softmax方法优化计算时用到的变量
    */
  //固定编码长度，2^30=1073741824是billion量级，足够公司任务使用
  private val maxCodeLength = 30



  /** 设置句子的最长长度，超过此长度的句子会被分割
    */
  def setMaxSentenceLength(maxSentenceLength: Int): this.type = {
    this.maxSentenceLength = maxSentenceLength
    this
  }

  def setVectorSize(vectorSize: Int): this.type = {
    this.vectorSize = vectorSize
    this
  }

  def setLearningRate(learningRate: Double): this.type = {
    this.learningRate = learningRate
    this
  }

  def setNumPartitions(numPartitions: Int): this.type = {
    this.numPartitions = numPartitions
    this
  }

  def setNumIterations(numIterations: Int): this.type = {
    this.numIterations = numIterations
    this
  }

  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  def setWindowSize(window: Int): this.type = {
    this.window = window
    this
  }

  def setMinCount(minCount: Int): this.type = {
    this.minCount = minCount
    this
  }

  def setAlgorithm(algorithm: Int): this.type = {
    this.algorithm = algorithm
    this
  }

  def setHsns(hsns: Int): this.type = {
    this.hsns = hsns
    this
  }

  def setNegative(negative: Int): this.type = {
    this.negative = negative
    this
  }

  /**
    * 计算sigmod值 存储到expTable
    * 将[-6,6]等距离划分成1000份
    * @return
    */
  //存储sigmod值得表的大小
  private val expTableSize = 1000
  //定义域长度[-6,6]
  private val maxExp = 6
  def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](expTableSize)
    var j = 0
    while (j < expTableSize) {
      val tmp = math.exp((2.0 * j / expTableSize - 1.0) * maxExp)
      expTable(j) = (tmp / (tmp + 1.0)).toFloat
      j += 1
    }
    expTable
  }

  /**根据训练集生成一些必须信息，如词库大小、单词的idyi映射等
    *
    * @param dataset 训练集
    */
  def copusPrepare[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    val words = dataset.flatMap(x => x)
    //将训练集文本转成词库
    corpus = words.map(w => (w, 1)).reduceByKey(_ + _).
      filter(_._2 >= minCount).
      map(x => CorpusWord(x._1,x._2,new Array[Int](maxCodeLength),new Array[Int](maxCodeLength),0)).
      collect().sortWith((a, b) => a.cn > b.cn)
    //词库大小
    corpusSize = corpus.length
    //生成词库单词与id的hash映射
    var i = 0
    while (i < corpusSize) {
      corpusHash += corpus(i).word -> i
      corpusCount += corpus(i).cn
      i += 1
    }
  }


  /**
    * 生成概率分布表
    * @param power
    */
  def createCumTable(power: Double = 0.75): Unit = {
    //分布表
    val tableSize = Int.MaxValue
    cumTable = new Array[Int](corpusSize)
    val corpusWordPowSum = corpus.map(x => math.pow(x.cn,power)).reduce(_+_).toFloat
    var cumulative = 0.0f
    var index = 0
    while (index < corpusSize) {
      cumulative += math.pow(corpus(index).cn,power).toFloat
      cumTable(index) = math.round(cumulative / corpusWordPowSum * tableSize)
    }
    if(cumTable.length>0) cumTable(cumTable.length-1) = tableSize
  }
  /**
    * 生成哈夫曼树
    */
  def createBinaryTree(): Unit = {
    val count = new Array[Long](corpusSize * 2 + 1)
    val binary = new Array[Int](corpusSize * 2 + 1)
    val parentNode = new Array[Int](corpusSize * 2 + 1)
    val code = new Array[Int](maxCodeLength)
    val point = new Array[Int](maxCodeLength)
    //初始化count数组，前corpusSize个数是权值，其余都是大数1e9
    var a = 0
    while (a < corpusSize) {
      count(a) = corpus(a).cn
      a += 1
    }
    while (a < 2 * corpusSize) {
      count(a) = 1e9.toInt
      a += 1
    }
    //设置两个指针，分别指向最后一个词及它的后一位
    var pos1 = corpusSize - 1
    var pos2 = corpusSize
    //构建哈夫曼树
    var min1i = 0
    var min2i = 0
    a = 0
    while (a < corpusSize - 1) {
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min1i = pos1
          pos1 -= 1
        } else {
          min1i = pos2
          pos2 += 1
        }
      } else {
        min1i = pos2
        pos2 += 1
      }
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min2i = pos1
          pos1 -= 1
        } else {
          min2i = pos2
          pos2 += 1
        }
      } else {
        min2i = pos2
        pos2 += 1
      }
      count(corpusSize + a) = count(min1i) + count(min2i)
      parentNode(min1i) = corpusSize + a
      parentNode(min2i) = corpusSize + a
      binary(min2i) = 1
      a += 1
    }
    //给每个词(即叶子节点)生成哈夫曼编码
    var i = 0
    a = 0
    while (a < corpusSize) {
      var b = a
      i = 0
      while (b != corpusSize * 2 - 2) {
        code(i) = binary(b)
        point(i) = b
        i += 1
        b = parentNode(b)
      }
      corpus(a).codeLen = i
      corpus(a).point(0) = corpusSize - 2
      b = 0
      while (b < i) {
        corpus(a).code(i - b - 1) = code(b)
        corpus(a).point(i - b) = point(b) - corpusSize
        b += 1
      }
      a += 1
    }
  }


  /**
    * 计算词向量表达
    * @param dataset 训练集
    * @return word2vec模型
    */
  def train[S <: Iterable[String]](dataset: RDD[S]): Word2VectorModel = {

    copusPrepare(dataset)
    if(hsns==0 && negative>0){
      createCumTable()
    }else{
      createBinaryTree()
    }
    val sc = dataset.sparkContext
    //将词库、hash映射、sigmod值表 等数据广播到各节点
    val expTable = sc.broadcast(createExpTable())
    val bcCorpus = sc.broadcast(corpus)
    val bcCorpusHash = sc.broadcast(corpusHash)
    val bcCumTable = sc.broadcast(cumTable)
    //将训练集word转成id,且根据
    val sentencesOrigin = dataset.mapPartitions{ sentenceIter =>
      sentenceIter.flatMap{ sentence =>
        val indexesFromSentence = sentence.flatMap(bcCorpusHash.value.get)
        indexesFromSentence.grouped(maxSentenceLength).map(_.toArray)
      }
    }.map(x=>(x.sum,x)).cache()
    val sentences = sentencesOrigin.
      partitionBy(new RangePartitioner(numPartitions,sentencesOrigin)).
      values.cache()
    sentencesOrigin.unpersist()
    //设置随机数生成器 借用spark原生随机数生成器 速度更快
    val initRandom = new XORShiftRandom(seed)
    //初始化词向量矩阵，initRandom.nextFloat()生成[0,1]中随机数，最终值范围是[-0.5/vectorsize , 0.5/vectorsize]
    val embeddingMatrix = Array.fill[Float](corpusSize*vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    //Hs的参数矩阵
    val weightMatrix = new Array[Float](corpusSize*vectorSize)
    //negative-sampleing的参数矩阵
    val weightMatrixNegative = new Array[Float](corpusSize*vectorSize)
    var alpha = learningRate

    for (k <- 1 to numIterations) {
      System.out.println(s"开始第${k}轮迭代")
      val bcEmbeddingMatrix = sc.broadcast(embeddingMatrix)
      val bcWeightMatrix = sc.broadcast(weightMatrix)
      val partial = sentences.mapPartitionsWithIndex { case (idx, iter) =>
        val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
        val syn0Modify = new Array[Int](corpusSize)
        val syn1Modify = new Array[Int](corpusSize)
        val model = iter.foldLeft((bcEmbeddingMatrix.value, bcEmbeddingMatrix.value, 0L, 0L)) {
          case ((syn0, syn1, lastWordCount, wordCount), sentence) =>
            var lwc = lastWordCount
            var wc = wordCount
            //动态更新学习率
            if (wordCount - lastWordCount > 10000) {
              lwc = wordCount
              alpha = learningRate * (1 - wordCount.toDouble / (numIterations * corpusCount + 1))
              if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
            }
            wc += sentence.length
            var pos = 0
            while (pos < sentence.length) {
              val word = sentence(pos)
              //滑动窗口为0-window间随机数
              val b = random.nextInt(window)
              var neu1 = new Array[Float](vectorSize)
              val neu1e = new Array[Float](vectorSize)
              var target = 0
              var label = 0
              if (algorithm == 1) {
                // 采用Skip-gram模型
                var a = b
                while (a < window * 2 + 1 - b) {
                  if (a != window) {
                    val c = pos - window + a
                    if (c >= 0 && c < sentence.length) {
                      //目标词的index
                      val lastWord = sentence(c)
                      val l1 = lastWord * vectorSize
                      //用 negative sampling训练
                      if (hsns == 0 && negative > 0) {
                        var d = 0
                        while (d < negative + 1) {
                          if (d == 0) {
                            target = word
                            label = 1
                          } else {
                            val sampleNumber = initRandom.nextInt(Int.MaxValue)
                            target = ArrayLikeHelper.bisect_left(bcCumTable.value, sampleNumber)
                            if(target == 0) target = sampleNumber % (corpusSize - 1) + 1
                            label = 0
                          }
                          if (d == 0 || target == word) {
                            val l2 = target * vectorSize
                            // 计算输入词向量与参数向量的内积
                            var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                            var g = 0.0f
                            if (f > maxExp) {
                              g = ((label - 1) * alpha).toFloat
                            } else if (f < -maxExp) {
                              g = ((label - 0) * alpha).toFloat
                            } else {
                              g = ((label - expTable.value(((f + maxExp) * (expTableSize / maxExp / 2.0)).toInt)) * alpha).toFloat
                            }
                            //更新参数向量
                            blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                            blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1)
                            syn1Modify(target) += 1
                          }
                          d += 1
                        }

                      } else { //用H-softmax训练
                        var d = 0
                        while (d < bcCorpus.value(word).codeLen) {
                          val inner = bcCorpus.value(word).point(d)
                          val l2 = inner * vectorSize
                          // 计算输入词向量与参数向量的内积
                          var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                          if (f > -maxExp && f < maxExp) {
                            f = expTable.value(((f + maxExp) * (expTableSize / maxExp / 2.0)).toInt)//等于求sigmod(f)
                            //计算负梯度 j=i时是1-f j<>i时是-f
                            val g = ((1 - bcCorpus.value(word).code(d) - f) * alpha).toFloat
                            //更新参数向量
                            blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                            blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1)
                            syn1Modify(inner) += 1
                          }
                          d += 1
                        }
                      }
                      //更新词向量
                      blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1)
                      syn0Modify(lastWord) += 1
                    }
                  }
                  a += 1
                }
              } else {  //采用cbow模型
                var a = b
                var cw = 0
                while (a < window * 2 + 1 -b) {
                  if (a != window){
                    val c = pos - window + a
                    if (c >= 0 && c < sentence.length) {
                      val lastWord = sentence(c)
                      val l1 = lastWord * vectorSize
                      blas.saxpy(vectorSize, 1.0f, syn0, l1, 1, neu1, 0, 1)
                      cw += 1
                    }
                  }
                  a += 1
                }
                if (cw > 0) {
                  neu1 = neu1.map(_ / cw)
                  //用 negative sampling训练
                  if (hsns == 0 && negative > 0) {
                    var d = 0
                    while (d < negative + 1) {
                      if (d == 0) {
                        target = word
                        label = 1
                      } else {
                        val sampleNumber = initRandom.nextInt(Int.MaxValue)
                        target = ArrayLikeHelper.bisect_left(bcCumTable.value, sampleNumber)
                        if(target == 0) target = sampleNumber % (corpusSize - 1) + 1
                        label = 0
                      }
                      if (d == 0 || target == word) {
                        val l2 = target * vectorSize
                        // 计算输入词向量与参数向量的内积
                        var f = blas.sdot(vectorSize, neu1, 0, 1, syn1, l2, 1)
                        var g = 0.0f
                        //目标似然函数最大化 计算正梯度的一部分 gradient = (label - sigmod(f)) * x
                        if (f > maxExp) {
                          g = ((label - 1) * alpha).toFloat
                        } else if (f < -maxExp) {
                          g = ((label - 0) * alpha).toFloat
                        } else {
                          g = ((label - expTable.value(((f + maxExp) * (expTableSize / maxExp / 2.0)).toInt)) * alpha).toFloat
                        }
                        //更新参数向量
                        blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                        blas.saxpy(vectorSize, g, neu1, 0, 1, syn1, l2, 1)
                        syn1Modify(target) += 1
                      }
                      d += 1
                    }
                  } else { //用H-softmax训练
                    var d = 0
                    while (d < bcCorpus.value(word).codeLen) {
                      val inner = bcCorpus.value(word).point(d)
                      val l2 = inner * vectorSize
                      // Propagate hidden -> output
                      var f = blas.sdot(vectorSize, neu1, 0, 1, syn1, l2, 1)
                      if (f > -maxExp && f < maxExp) {
                        f = expTable.value(((f + maxExp) * (expTableSize / maxExp / 2.0)).toInt)
                        //等于求sigmod(f)
                        //计算负梯度 j=i时是1-f j<>i时是-f
                        val g = ((1 - bcCorpus.value(word).code(d) - f) * alpha).toFloat
                        //更新参数向量
                        blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                        blas.saxpy(vectorSize, g, neu1, 0, 1, syn1, l2, 1)
                        syn1Modify(inner) += 1
                      }
                      d += 1
                    }
                  }
                  //更新词向量
                  a = b
                  while (a < window * 2 + 1 -b) {
                    if (a != window){
                      val c = pos - window + a
                      if (c >= 0 && c < sentence.length) {
                        val lastWord = sentence(c)
                        val l1 = lastWord * vectorSize
                        blas.saxpy(vectorSize, 1.0f / cw, neu1e, 0, 1, syn0, l1, 1)
                        syn0Modify(lastWord) += 1
                      }
                    }
                    a += 1
                  }
                }
              }
              pos += 1
            }
            (syn0, syn1, lwc, wc)
        }
        val embeddingMatrixLocal = model._1
        val weightMatrixLocal = model._2
        // 将迭代过的词向量与参数向量输出
        Iterator.tabulate(corpusSize) { index =>
          if (syn0Modify(index) > 0) {
            Some((index, embeddingMatrixLocal.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(corpusSize) { index =>
          if (syn1Modify(index) > 0) {
            Some((index + corpusSize, weightMatrixLocal.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten
      }
      //合并各分区参数更新数据
//      val synAgg = partial.reduceByKey { case (v1, v2) =>
//        blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1)
//        v1
//      }.collect()
      //合并各分区参数更新数据
      val synAgg = partial.groupByKey(numPartitions).mapValues{ iter =>
        val iterSize = iter.size
        val arr = iter.toArray
        arr(initRandom.nextInt(iterSize))
      }.collect()
      //将更新后的词向量 参数向量赋予全局词向量 参数向量
      var i = 0
      while (i < synAgg.length) {
        val index = synAgg(i)._1
        if (index < corpusSize) {
          Array.copy(synAgg(i)._2, 0, embeddingMatrix, index * vectorSize, vectorSize)
        } else {
          Array.copy(synAgg(i)._2, 0, weightMatrix, (index - corpusSize) * vectorSize, vectorSize)
        }
        i += 1
      }
      //销毁广播变量
      bcEmbeddingMatrix.destroy()
      bcWeightMatrix.destroy()
    }
    sentences.unpersist()
    expTable.destroy()
    bcCorpus.destroy()
    bcCorpusHash.destroy()

    val wordArray = corpus.map(_.word)
    new Word2VectorModel(wordArray.zipWithIndex.toMap, embeddingMatrix)
  }


}


/**
import com.bj58.business.recsys.model.word2vec.Word2Vector
import com.bj58.business.recsys.model.word2vec.Word2VectorModel
  */


object Test{
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.
      builder().
      appName("esc-analyse").
      enableHiveSupport().
      getOrCreate()
    val sc = spark.sparkContext

    val path = "/home/hdp_lbg_ectech/resultdata/strategy/ads/ningshiqi_test/hy_cutword/102/20180801"
    val Array(data,data1) = sc.textFile(path).map(_.split("_").toIterable).
      randomSplit(Array(0.3,0.7))
    val hashIndex = data.flatMap(x=>x).map((_,1)).reduceByKey(_+_).collect().sortBy(-_._2).map(_._1).zipWithIndex
    var corpusHash = mutable.HashMap.empty[String, Int]
    var i =0
    while(i < hashIndex.length){
      val (word,ind) = hashIndex(i)
      corpusHash += word -> ind
      i += 1
    }
    val bcCorpusHash = sc.broadcast(corpusHash)
    val df = data.mapPartitions{ senIter =>
      senIter.map(sen => sen.flatMap(bcCorpusHash.value.get).toArray)
    }.cache()

    val word2Vec = new Word2Vector().setNumPartitions(100).setNumIterations(50)
    val model = word2Vec.train(data)
    val vectors = model.getVectors
    model.findSynonyms("违章服务",10)
//    val seed = 1234
//    val initRandom = new XORShiftRandom(seed)
//    for( i <- 1 to 100){
//      println(initRandom.nextInt(10))
//    }

    val df1 = spark.sql("""select tag from hdp_lbg_ectech_ads.esc_ads""").rdd.map{ x=>
      val obj = JSON.parseObject(x.getString(0))
      var lx = ""
      if(obj.containsKey("2501101")){
        lx = obj.getString("2501101")
      }
      var pp = ""
      if(obj.containsKey("2501107")){
        pp = obj.getString("2501107")
      }
      var cx = ""
      if(obj.containsKey("2501108")){
        cx = obj.getString("2501108")
      }
      (lx,pp,cx)
    }.cache()

  }
}



