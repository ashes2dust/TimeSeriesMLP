package org.apache.spark.ml.timeseries

import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import org.apache.spark.ml.ann.HCFeedForwardTopology
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.regression.MultilayerPerceptronRegressor
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

private[timeseries] trait TimeSeriesMLPParams extends Params
  with HasTol with HasSeed with HasMaxIter with HasStepSize
  with HasSolver {

  import TimeSeriesMLP._

  final val valueCol: Param[String] = new Param[String](this, "valueCol",
    "The value column")

  def getValueCol: String = $(valueCol)

  final val tsCol: Param[String] = new Param[String](this, "tsCol",
    "The time series column")

  def getTsCol: String = $(tsCol)

  final val initialWeights: Param[Vector] = new Param[Vector](this, "initialWeights",
    "The initial weights of model")

  final val pattern: Param[String] = new Param[String](this, "pattern",
    "The timestamp pattern")

  def getInitialWeights: Vector = $(initialWeights)

  final val blockSize: Param[Int] = new Param[Int](this, "blockSize",
    "Block size", ParamValidators.gt(0))

  def getBlockSize: Int = $(blockSize)

  final val windowSize: Param[Int] = new Param[Int](this, "windowSize",
  "The window size", ParamValidators.gt(0))

  def getWindowSize: Int = $(windowSize)

  final val hiddenLayers: IntArrayParam = new IntArrayParam(this, "layers",
    "Size of layers",
    (t: Array[Int]) => t.forall(ParamValidators.gt(0)) && t.length >= 1)

  def getHiddenLayers: Array[Int] = $(hiddenLayers)

  final val activation: Param[String] = new Param[String](this, "activation",
    "The activation function of hidden layers",
    ParamValidators.inArray(Array("relu", "sigmoid", "tanh", "identity")))

  def getActivation: String = $(activation)

  final override val solver: Param[String] = new Param[String](this, "solver",
    "The solver algorithm for optimization. Supported options: " +
      s"${supportedSolvers.mkString(", ")}. (Default l-bfgs)",
    ParamValidators.inArray[String](supportedSolvers))

  final val predictDays: IntParam = new IntParam(this, "predictDays",
  "The number of days to forecast", ParamValidators.gt(0))

  def getPredictDays: Int = $(predictDays)

  setDefault(
    maxIter -> 100,
    tol -> 1e-6,
    blockSize -> 128,
    solver -> TimeSeriesMLP.LBFGS,
    stepSize -> 0.03,
    valueCol -> "value",
    tsCol -> "timestamp",
    pattern -> "dd-MM-yy",
    windowSize -> 1,
    activation -> "relu",
    predictDays -> 1
  )

}

object TimeSeriesMLP extends DefaultParamsReadable[TimeSeriesMLP] {
  private[timeseries] val LBFGS = "l-bfgs"

  private[timeseries] val GD = "gd"

  private[timeseries] val supportedSolvers = Array(LBFGS, GD)

  override def load(path: String): TimeSeriesMLP = super.load(path)

  def slidingWindowTransform(data: RDD[(Double, Long)], windowSize: Int): RDD[(Vector, Double)] = {
    val windowData = data.flatMap { case (v, id) =>
      var i = id
      val arr = new ListBuffer[(Long, (Long, Double))]()
      while (i >= id - windowSize) {
        val tp = Tuple2(id, v)
        arr += Tuple2(i, tp)
        i -= 1
      }
      arr
    }.groupByKey()
      .map { x =>
        val arr = x._2.toArray.sorted.map(_._2)
        (x._1, (Vectors.dense(arr.take(arr.length - 1)), arr.last))
      }.sortBy(_._1)
      .map(_._2).filter(_._1.size == windowSize)
    windowData
  }

  // DF('value', 'timestamp') -> DF('features', 'label')
  def slidingWindowTransform(
      dataset: Dataset[_],
      windowSize: Int,
      sort: Boolean = false,
      valueCol: String = "value",
      tsCol: String = "timestamp",
      pattern: String = "dd-MM-yy",
      featuresCol: String = "features",
      labelCol: String = "label"): DataFrame = {

    val data = dataset.select(valueCol, tsCol)
      .rdd.map { case Row(value: String, ts: String) =>
      val format = new SimpleDateFormat(pattern)
      (value.toDouble, format.parse(ts).getTime)
    }

    val sortedData = {
      if (sort)
        data.sortBy(_._2).zipWithIndex().map(x => (x._1._1, x._2))
      else
        data.zipWithIndex().map(x => (x._1._1, x._2))
    }

    val spark = dataset.sparkSession

    import spark.implicits._

    slidingWindowTransform(sortedData, windowSize).toDF(featuresCol, labelCol)
  }

}

class TimeSeriesMLP(override val uid: String) extends Estimator[TimeSeriesMLPModel]
  with TimeSeriesMLPParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("tsmlp"))

  def setTol(value: Double): this.type = set(tol, value)

  def setValueCol(value: String): this.type = set(valueCol, value)

  def setTsCol(value: String): this.type = set(tsCol, value)

  def setHiddenLayers(value: Array[Int]): this.type = set(hiddenLayers, value)

  def setWindowSize(value: Int): this.type = set(windowSize, value)

  def setPattern(value: String): this.type = set(pattern, value)

  def setActivation(value: String): this.type = set(activation, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setStepSize(value: Double): this.type = set(stepSize, value)

  def setSolver(value: String): this.type = set(solver, value)

  def setInitialWeights(value: Vector): this.type = set(initialWeights, value)

  def setBlockSize(value: Int): this.type = set(blockSize, value)

  override def copy(extra: ParamMap): TimeSeriesMLP = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    // TODO
    schema
  }

  override def fit(dataset: Dataset[_]): TimeSeriesMLPModel = {

    val df = TimeSeriesMLP.slidingWindowTransform(dataset, $(windowSize), sort=true,
      $(valueCol), $(tsCol), $(pattern))

    val layers = Array($(windowSize)) ++ $(hiddenLayers) ++ Array(1)
    val trainer = new MultilayerPerceptronRegressor()
      .setLayers(layers)
      .setActivation($(activation))
      .setBlockSize($(blockSize))
      .setMaxIter($(maxIter))
      .setSeed($(seed))
      .setSolver($(solver))
      .setStepSize($(stepSize))

    if (isDefined(initialWeights)) {
      trainer.setInitialWeights($(initialWeights))
    }

    val mlpModel = trainer.fit(df)

    copyValues(new TimeSeriesMLPModel(uid, layers, mlpModel.weights, $(activation)))
  }
}

class TimeSeriesMLPModel(
                          override val uid: String,
                          val layers: Array[Int],
                          val weights: Vector,
                          val activationType: String) extends Model[TimeSeriesMLPModel] with TimeSeriesMLPParams {

  private[ml] val mlpModel = HCFeedForwardTopology
    .multiLayerRegressionPerceptron(layers, activationType)
    .model(weights)

  def setValueCol(value: String): this.type = set(valueCol, value)

  def setTsCol(value: String): this.type = set(tsCol, value)

  def setPattern(value: String): this.type = set(pattern, value)

  def setPredictDays(value: Int): this.type = set(predictDays, value)

  override def copy(extra: ParamMap): TimeSeriesMLPModel = {
    val copied = new TimeSeriesMLPModel(uid, layers, weights, activationType)
    copied.copyValues(copied, extra)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    // TODO
    dataset.toDF()
  }

  override def transformSchema(schema: StructType): StructType = {
    // TODO
    schema
  }

  def predict(features: Vector): Double = {
    val v = mlpModel.predict(features)
    v(0)
  }

  case class TSData(value: Double, timestamp: String)

  // Forecast the future days according to predictDays
  def predict(dataset: Dataset[_]): DataFrame = {
    val format = new SimpleDateFormat($(pattern))

    val toTime = dataset.sparkSession.udf.register("toTime", (ts: String) => format.parse(ts).getTime)

    val rawDF = dataset.toDF($(valueCol), $(tsCol))
    val df = rawDF.withColumn($(tsCol), toTime(rawDF($(tsCol))))
      .withColumn($(valueCol), rawDF($(valueCol)).cast(DoubleType)).sort($(tsCol)).cache()

    val histories = df.rdd.map { case Row(value: Double, ts: Long) =>
      (value, ts)
    }.collect()

    if (histories.length < $(windowSize)) {
      throw new IllegalArgumentException(
        "The dataset size should not less than window size."
      )
    }

    // Forecast iteratively
    var feature = histories.map(_._1).drop(histories.length - $(windowSize))
    val predictions = ArrayBuffer[Double]()

    for (i <- 0 until $(predictDays)) {
      val pred = predict(new DenseVector(feature))
      predictions += pred
      feature = feature.drop(1) :+ pred
    }

    // Transform predictions to DF["value", "ts"]
    val startDate = new Date(histories.last._2)
    val cal = Calendar.getInstance()

    cal.setTime(startDate)

    val futures = ArrayBuffer[TSData]()
    for (pred <- predictions) {
      cal.roll(Calendar.DATE, true)
      futures += TSData(pred, format.format(cal.getTime))
    }

    df.unpersist()

    dataset.sparkSession.createDataFrame(futures)
  }
}
