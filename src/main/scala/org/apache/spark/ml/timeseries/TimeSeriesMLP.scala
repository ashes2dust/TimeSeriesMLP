package org.apache.spark.ml.timeseries

import java.text.SimpleDateFormat

import org.apache.spark.ml.ann.HCFeedForwardTopology
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.regression.MultilayerPerceptronRegressor
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

private[timeseries] trait TimeSeriesMLPParams extends Params
  with HasTol with HasSeed with HasMaxIter with HasStepSize with HasSolver {

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

  setDefault(
    maxIter -> 100,
    tol -> 1e-6,
    blockSize -> 128,
    solver -> TimeSeriesMLP.LBFGS,
    stepSize -> 0.03,
    valueCol -> "value",
    tsCol -> "timestamp",
    pattern -> "dd-MM-yy",
    windowSize -> 1
  )

}

class TimeSeriesMLPModel(
    override val uid: String,
    val layers: Array[Int],
    val weights: Vector) extends Model[TimeSeriesMLPModel] with TimeSeriesMLPParams {

  private[ml] val mlpModel = HCFeedForwardTopology
    .multiLayerRegressionPerceptron(layers)
    .model(weights)

  override def copy(extra: ParamMap): TimeSeriesMLPModel = {
    val copied = new TimeSeriesMLPModel(uid, layers, weights)
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
}

private[timeseries] object TimeSeriesMLP extends DefaultParamsReadable[TimeSeriesMLP] {
  private[timeseries] val LBFGS = "l-bfgs"

  private[timeseries] val GD = "gd"

  private[timeseries] val supportedSolvers = Array(LBFGS, GD)

  override def load(path: String): TimeSeriesMLP = super.load(path)
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

  def slidingWindowTransform(data: RDD[(Double, Long)], windowSize: Int): RDD[(Vector, Double)] = {
    val windowData = data.flatMap { case (v, id) =>
      var i = id
      val arr = new ListBuffer[(Long, mutable.Iterable[(Long, Double)])]()
      while (i >= id - windowSize) {
        val tp = mutable.Iterable(Tuple2(id, v))
        arr += Tuple2(i, tp)
        i -= 1
      }
      arr
    }.reduceByKey(_ ++ _)
      .map { x =>
        val arr = x._2.toArray.sorted.map(_._2)
        (x._1, (Vectors.dense(arr.take(arr.length - 1)), arr.last))
      } // TODO: Should I sort rdd by timestamp ? I can optimize this.
      .map(_._2).filter(_._1.size == windowSize)
    windowData
  }

  override def fit(dataset: Dataset[_]): TimeSeriesMLPModel = {
    // Preprocessing
    def toTimeStamp(str: String): Long = {
      val format = new SimpleDateFormat($(pattern))
      format.parse(str).getTime
    }

    // cast string to timestamp
    val spark = dataset.sparkSession
    val toTime = spark.udf.register("toTime", (str: String) => toTimeStamp(str))
    val rawData = dataset
      .withColumn("timestamp", toTime(dataset(getTsCol)))
      .select("value", "timestamp")
      .rdd.map(row => (row.getString(0).toDouble, row.getLong(1)))
      .sortBy(_._2).zipWithIndex().map(x => (x._1._1, x._2))

    // Double -> (Vector, Double)
    val data = slidingWindowTransform(rawData, $(windowSize))

    // 构造MLP，修改最后一层，不是Sigmoid，当然也可以重新实现multiLayerPerceptron函数
    val layers = Array($(windowSize)) ++ $(hiddenLayers) ++ Array(1)
    val trainer = new MultilayerPerceptronRegressor()
        .setLayers(layers)
        .setBlockSize($(blockSize))
        .setMaxIter($(maxIter))
        .setSeed($(seed))
        .setSolver($(solver))
        .setStepSize($(stepSize))

    if (isDefined(initialWeights)) {
      trainer.setInitialWeights($(initialWeights))
    }

    import spark.implicits._

    val df = data.toDF("features", "label")

    val mlpModel = trainer.fit(df)

    copyValues(new TimeSeriesMLPModel(uid, layers, mlpModel.weights))
  }
}

