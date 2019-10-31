package org.apache.spark.ml.timeseries

import java.text.SimpleDateFormat

import org.apache.spark.ml.ann.{FeedForwardTrainer, HCFeedForwardTopology, SimpleLayerWithSquaredError, TopologyModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

private[timeseries] trait TimeSeriesMLPParams extends Params
  with HasTol with HasInputCol with HasOutputCol with HasSeed
  with HasMaxIter with HasStepSize with HasSolver {

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

  // TODO: input layer (layers.head) is the window size,
  // and output layer (layers.tail) should be 1 in time series.
  final val layers: IntArrayParam = new IntArrayParam(this, "layers",
    "Size of layers",
    (t: Array[Int]) => t.forall(ParamValidators.gt(0)) && t.length > 1)

  final override val solver: Param[String] = new Param[String](this, "solver",
    "The solver algorithm for optimization. Supported options: " +
      s"${supportedSolvers.mkString(", ")}. (Default l-bfgs)",
    ParamValidators.inArray[String](supportedSolvers))

  def getLayers(): Array[Int] = $(layers)

  setDefault(
    maxIter -> 100,
    tol -> 1e-6,
    blockSize -> 128,
    solver -> TimeSeriesMLP.LBFGS,
    stepSize -> 0.03,
    valueCol -> "value",
    tsCol -> "timestamp",
    pattern -> "dd-MM-yy"
  )

}

class TimeSeriesMLPModel(override val uid: String,
                         val layers: Array[Int],
                         val weights: Vector) extends Model[TimeSeriesMLPModel] {

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

  def setLayers(value: Array[Int]): this.type = set(layers, value)

  def setPattern(value: String): this.type = set(pattern, value)

  //  def setInputCol(value: String): this.type = set(inputCol, value)
  //
  //  def setOutputCol(value: String): this.type = set(outputCol, value)

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

  def slidingWindowTransform(data: RDD[(Double, Long)], windowSize: Int): RDD[(Vector, Vector)] = {
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
        (x._1, (Vectors.dense(arr.take(arr.length - 1)), Vectors.dense(arr.last)))
      } // TODO: Should I sort rdd by timestamp ? I can optimize this.
      .map(_._2).filter(_._1.size == windowSize)
    windowData
  }

  override def fit(dataset: Dataset[_]): TimeSeriesMLPModel = {
    // Preprocessing
    def toTimeStamp(str: String): Long = {
      val format = new SimpleDateFormat($(pattern))
      //      new Timestamp(format.parse(str).getTime)
      format.parse(str).getTime
    }

    // cast string to timestamp
    val spark = dataset.sparkSession
    val toTime = spark.udf.register("toTime", (str: String) => toTimeStamp(str))
    val rawData = dataset
      .withColumn("timestamp", toTime(dataset(getTsCol)))
      //        .withColumn("value", dataset(getInputCol).cast(DoubleType))
      .select("value", "timestamp")
      .rdd.map(row => (row.getString(0).toDouble, row.getLong(1)))
      .sortBy(_._2).zipWithIndex().map(x => (x._1._1, x._2))


    rawData.take(10).foreach(println)

    // 构造MLP，修改最后一层，不是Sigmoid，当然也可以重新实现multiLayerPerceptron函数
    val topology = HCFeedForwardTopology.multiLayerRegressionPerceptron(getLayers())

    val trainer = new FeedForwardTrainer(topology, getLayers().head, getLayers().last)
    if (isDefined(initialWeights)) {
      trainer.setWeights($(initialWeights))
    } else {
      trainer.setSeed($(seed))
    }

    if ($(solver) == TimeSeriesMLP.LBFGS) {
      trainer.LBFGSOptimizer
        .setConvergenceTol($(tol))
        .setNumIterations($(maxIter))
    } else if ($(solver) == TimeSeriesMLP.GD) {
      trainer.SGDOptimizer
        .setNumIterations($(maxIter))
        .setConvergenceTol($(tol))
        .setStepSize($(stepSize))
    } else {
      throw new IllegalArgumentException(
        s"The solver $solver is not supported by MultilayerPerceptronClassifier.")
    }

    // TODO: Transform data using sliding window.
    // Double -> (Vector, Double)
    val windowSize = getLayers().head
    //    val rawData: RDD[(Double, Long)] = null // TODO
    val data: RDD[(Vector, Vector)] = slidingWindowTransform(rawData, windowSize)
    data.take(10).foreach(println)

    trainer.setStackSize($(blockSize))
    val mlpModel = trainer.train(data)

    println(mlpModel.weights)
    new TimeSeriesMLPModel(uid, getLayers(), mlpModel.weights)
  }
}

