package com.bj58.business.recsys.model.mf.als

import java.util

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.mllib.linalg.CholeskyDecomposition

/**
  *
  *
  * @Author: renjian01
  * @Creation: 2018/12/27 18:21
  */


/**
  * Representing a normal equation to solve the following weighted least squares problem:
  *
  * minimize \sum,,i,, c,,i,, (a,,i,,^T^ x - b,,i,,)^2^ + lambda * x^T^ x.
  *
  * Its normal equation is given by
  *
  * \sum,,i,, c,,i,, (a,,i,, a,,i,,^T^ x - b,,i,, a,,i,,) + lambda * x = 0.
  */
class NormalEquation(val k: Int) extends Serializable {

  /** Number of entries in the upper triangular part of a k-by-k matrix. */
  val triK = k * (k + 1) / 2
  /** A^T^ * A */
  val ata = new Array[Double](triK)
  /** A^T^ * b */
  val atb = new Array[Double](k)

  private val da = new Array[Double](k)
  private val upper = "U"

  private def copyToDouble(a: Array[Float]): Unit = {
    var i = 0
    while (i < k) {
      da(i) = a(i)
      i += 1
    }
  }

  /** Adds an observation. */
  def add(a: Array[Float], b: Double, c: Double = 1.0): this.type = {
    require(c >= 0.0)
    require(a.length == k)
    copyToDouble(a)
    blas.dspr(upper, k, c, da, 1, ata)
    if (b != 0.0) {
      blas.daxpy(k, c * b, da, 1, atb, 1)
    }
    this
  }

  /** Merges another normal equation object. */
  def merge(other: NormalEquation): this.type = {
    require(other.k == k)
    blas.daxpy(ata.length, 1.0, other.ata, 1, ata, 1)
    blas.daxpy(atb.length, 1.0, other.atb, 1, atb, 1)
    this
  }

  /** Resets everything to zero, which should be called after each solve. */
  def reset(): Unit = {
    util.Arrays.fill(ata, 0.0)
    util.Arrays.fill(atb, 0.0)
  }
}


trait LeastSquareSolver extends Serializable {

  def solve(ne: NormalEquation, lambda: Double): Array[Float]

}


class CholeskySolver extends LeastSquareSolver {

  /**
    * Solves a least squares problem with L2 regularization:
    *
    *   min norm(A x - b)^2^ + lambda * norm(x)^2^
    *
    * @param ne a [[NormalEquation]] instance that contains AtA, Atb, and n (number of instances)
    * @param lambda regularization constant
    * @return the solution x
    */
  override def solve(ne: NormalEquation, lambda: Double): Array[Float] = {
    val k = ne.k
    // Add scaled lambda to the diagonals of AtA.
    var i = 0
    var j = 2
    while (i < ne.triK) {
      ne.ata(i) += lambda
      i += j
      j += 1
    }
   // CholeskyDecomposition.solve(ne.ata, ne.atb)
    val x = new Array[Float](k)
    i = 0
    while (i < k) {
      x(i) = ne.atb(i).toFloat
      i += 1
    }
    ne.reset()
    x
  }
}

