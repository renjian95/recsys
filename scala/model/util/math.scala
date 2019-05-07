package com.bj58.business.recsys.model.util

import breeze.linalg.{DenseVector, SparseVector, Vector}
import scala.math._
import com.github.fommil.netlib.{BLAS => NetlibBLAS, F2jBLAS}

/**
  * @Author: renjian01
  * @Creation: 2019/4/7 14:31
  */

object math extends Serializable {


  @transient private var _f2jBLAS: NetlibBLAS = _

  private def f2jBLAS: NetlibBLAS = {
    if (_f2jBLAS == null) {
      _f2jBLAS = new F2jBLAS
    }
    _f2jBLAS
  }

  /**
    * 数值稳定的log1p(exp(x))函数
    * @param x
    * @return
    */
  def log1pExp(x: Double): Double = {
    if (x > 0) x + scala.math.log1p(exp(-x)) else log1p(exp(x))
  }

  def scal(a: Double, v: Vector[Double]): Unit = {
    v match {
      case sx: SparseVector[Double] =>
        f2jBLAS.dscal(sx.data.length, a, sx.data, 1)
      case dx: DenseVector[Double] =>
        f2jBLAS.dscal(dx.data.length, a, dx.data, 1)
      case _ =>
        throw new IllegalArgumentException(s"scal doesn't support vector type ${v.getClass}.")
    }
  }

}
