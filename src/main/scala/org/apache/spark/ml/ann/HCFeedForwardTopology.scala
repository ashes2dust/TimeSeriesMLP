package org.apache.spark.ml.ann


private[ann] class ReluFunction extends ActivationFunction {

  override def eval: Double => Double = x => math.max(0.0, x)

  override def derivative: Double => Double = z => if (z == 0.0) 0.0 else 1.0
}

private[ann] class IdentityFunction extends ActivationFunction {

  override def eval: Double => Double = x => x

  override def derivative: Double => Double = _ => 1.0
}

private[ann] class TanhFunction extends ActivationFunction {

  override def eval: Double => Double = x => 2 / (1 + math.exp(-2 * x)) - 1

  override def derivative: Double => Double = z => 1 - z * z
}

private[ml] object HCFeedForwardTopology {
  /**
    * Creates a multi-layer regression perceptron
    *
    * @param layerSizes sizes of layers including input and output size
    * @return multilayer perceptron topology
    */
  def multiLayerRegressionPerceptron(
      layerSizes: Array[Int],
      activation: String = "relu"): FeedForwardTopology = {
    val layers = new Array[Layer]((layerSizes.length - 1) * 2)
    for (i <- 0 until layerSizes.length - 1) {
      layers(i * 2) = new AffineLayer(layerSizes(i), layerSizes(i + 1))
      layers(i * 2 + 1) =
        if (i == layerSizes.length - 2) {
          new SimpleLayerWithSquaredError()
        } else {
          activation match {
            case "sigmoid" => new FunctionalLayer(new SigmoidFunction)
            case "identity" => new FunctionalLayer(new IdentityFunction)
            case "tanh" => new FunctionalLayer(new TanhFunction)
            case _ => new FunctionalLayer(new ReluFunction)
          }
        }
    }
    FeedForwardTopology(layers)
  }
}
