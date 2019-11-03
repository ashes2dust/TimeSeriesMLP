package org.apache.spark.ml.regression

import org.apache.spark.internal.Logging
import org.apache.spark.ml.ann.{FeedForwardTrainer, HCFeedForwardTopology}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}

private[regression] trait MultilayerPerceptronRegressorParams extends Params
  with HasTol with HasSeed with HasMaxIter with HasStepSize with HasSolver
  with HasPredictionCol {

  import MultilayerPerceptronRegressor._

  final val initialWeights: Param[Vector] = new Param[Vector](this, "initialWeights",
    "The initial weights of model")

  def getInitialWeights: Vector = $(initialWeights)

  final val blockSize: Param[Int] = new Param[Int](this, "blockSize",
    "Block size", ParamValidators.gt(0))

  def getBlockSize: Int = $(blockSize)

  final val layers: IntArrayParam = new IntArrayParam(this, "layers",
    "Size of layers",
    (t: Array[Int]) => t.forall(ParamValidators.gt(0)) && t.length > 1)

  def getLayers: Array[Int] = $(layers)

  final val activation: Param[String] = new Param[String](this, "activation",
  "The activation function of hidden layers",
    ParamValidators.inArray(Array("relu", "sigmoid", "tanh", "identity")))

  def getActivation: String = $(activation)

  final override val solver: Param[String] = new Param[String](this, "solver",
    "The solver algorithm for optimization. Supported options: " +
      s"${supportedSolvers.mkString(", ")}. (Default l-bfgs)",
    ParamValidators.inArray[String](supportedSolvers))

  setDefault(
    maxIter -> 100,
    tol -> 1e-6,
    blockSize -> 128,
    solver -> LBFGS,
    stepSize -> 0.03,
    activation -> "relu"
  )

}

private[regression] object MultilayerPerceptronRegressor extends DefaultParamsReadable[MultilayerPerceptronRegressor] {

  private[regression] val LBFGS = "l-bfgs"

  private[regression] val GD = "gd"

  private[regression] val supportedSolvers = Array(LBFGS, GD)

  override def load(path: String): MultilayerPerceptronRegressor = super.load(path)
}

class MultilayerPerceptronRegressor(override val uid: String)
  extends Regressor[Vector, MultilayerPerceptronRegressor, MultilayerPerceptronRegressorModel]
  with MultilayerPerceptronRegressorParams with DefaultParamsWritable with Logging {

  def this() = this(Identifiable.randomUID("mlpReg"))

  def setSeed(value: Long): this.type = set(seed, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setStepSize(value: Double): this.type = set(stepSize, value)

  def setSolver(value: String): this.type = set(solver, value)

  def setInitialWeights(value: Vector): this.type = set(initialWeights, value)

  def setBlockSize(value: Int): this.type = set(blockSize, value)

  def setLayers(value: Array[Int]): this.type = set(layers, value)

  def setActivation(value: String): this.type = set(activation, value)

  override def copy(extra: ParamMap): MultilayerPerceptronRegressor = defaultCopy(extra)

  def transformDataset(dataset: Dataset[_]): RDD[(Vector, Vector)] = {
    dataset.toDF().select($(featuresCol), $(labelCol))
      .rdd.map { case Row(in: Vector, out: Double) => (in, Vectors.dense(out))}
  }

  override protected def train(dataset: Dataset[_]): MultilayerPerceptronRegressorModel = {
    // TODO: Validate dataset

    val data = transformDataset(dataset)

    val topology = HCFeedForwardTopology.multiLayerRegressionPerceptron(getLayers, $(activation))

    val trainer = new FeedForwardTrainer(topology, getLayers.head, getLayers.last)
    if (isDefined(initialWeights)) {
      trainer.setWeights($(initialWeights))
    } else {
      trainer.setSeed($(seed))
    }

    if ($(solver) == MultilayerPerceptronRegressor.LBFGS) {
      trainer.LBFGSOptimizer
        .setConvergenceTol($(tol))
        .setNumIterations($(maxIter))
    } else if ($(solver) == MultilayerPerceptronRegressor.GD) {
      trainer.SGDOptimizer
        .setNumIterations($(maxIter))
        .setConvergenceTol($(tol))
        .setStepSize($(stepSize))
    } else {
      throw new IllegalArgumentException(
        s"The solver $solver is not supported by MultilayerPerceptronClassifier.")
    }

    trainer.setStackSize($(blockSize))

    val mlpModel = trainer.train(data)

    copyValues(new MultilayerPerceptronRegressorModel(this.uid, getLayers, mlpModel.weights, $(activation)))
  }
}

private[regression] class MultilayerPerceptronRegressorModel(
    override val uid: String,
    val layerSizes: Array[Int],
    val weights: Vector,
    val activaionType: String) extends RegressionModel[Vector, MultilayerPerceptronRegressorModel]
  with MultilayerPerceptronRegressorParams {

  private[ml] val mlpModel = HCFeedForwardTopology
    .multiLayerRegressionPerceptron(layerSizes, activaionType)
    .model(weights)

  override def copy(extra: ParamMap): MultilayerPerceptronRegressorModel = {
    defaultCopy(extra)
  }

  override def predict(features: Vector): Double = {
    mlpModel.predict(features)(0)
  }

  def predict(dataset: Dataset[_]): DataFrame = {
    // TODO: Validate dataset
    val df = dataset.toDF()

    val spark = dataset.sparkSession

    val predictUDF = spark.udf.register("prediction", (features: Vector) => predict(features))

    df.withColumn($(predictionCol), predictUDF(df($(featuresCol))))
  }
}