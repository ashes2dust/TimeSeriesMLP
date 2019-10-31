package org.apache.spark.ml.ann

object HCFeedForwardTopology {
  /**
   * Creates a multi-layer regression perceptron
   *
   * @param layerSizes sizes of layers including input and output size
   * @return multilayer perceptron topology
   */
  def multiLayerRegressionPerceptron(
    layerSizes: Array[Int]): FeedForwardTopology = {
    val layers = new Array[Layer]((layerSizes.length - 1) * 2)
    for (i <- 0 until layerSizes.length - 1) {
      layers(i * 2) = new AffineLayer(layerSizes(i), layerSizes(i + 1))
      layers(i * 2 + 1) =
        if (i == layerSizes.length - 2) {
          new SimpleLayerWithSquaredError()
        } else {
          new FunctionalLayer(new SigmoidFunction())
        }
    }
    FeedForwardTopology(layers)
  }
}
