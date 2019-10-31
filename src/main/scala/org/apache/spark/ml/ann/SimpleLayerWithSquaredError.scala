package org.apache.spark.ml.ann

import breeze.linalg.{sum => Bsum, DenseMatrix => BDM, DenseVector => BDV}
import java.util.Random


private[ann] class SimpleLayerWithSquaredError extends Layer {
  override val weightSize = 0
  override val inPlace = true

  override def getOutputSize(inputSize: Int): Int = inputSize

  override def createModel(weights: BDV[Double]): LayerModel =
    new SimpleLayerModelWithSquaredError()

  override def initModel(weights: BDV[Double], random: Random): LayerModel =
    new SimpleLayerModelWithSquaredError()
}

private[ann] class SimpleLayerModelWithSquaredError extends LayerModel with LossFunction {
  override def loss(output: BDM[Double], target: BDM[Double], delta: BDM[Double]): Double = {
    ApplyInPlace(output, target, delta, (o: Double, t: Double) => o - t)
    val error = Bsum(delta *:* delta) / 2 / output.cols
    error
  }

  // loss layer models do not have weights
  override val weights: BDV[Double] = new BDV[Double](0)

  override def eval(data: BDM[Double], output: BDM[Double]): Unit = {
    ApplyInPlace(data, output, (x: Double) => x)
  }

  override def computePrevDelta(delta: BDM[Double], output: BDM[Double], prevDelta: BDM[Double]): Unit = {
    /* loss layer model computes delta in loss function */
  }

  override def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit = {
    /* loss layer model does not have weights */
  }
}