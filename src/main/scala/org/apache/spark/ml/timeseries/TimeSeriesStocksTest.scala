package org.apache.spark.ml.timeseries

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.Vector

object TimeSeriesStocksTest {
  var input = "data/sp500_train.csv"
  var testInput = "data/sp500_test.csv"
  var windowSize = 50
  var pattern = "dd-MM-yy"
  var hiddenLayers = Array(100, 100)
  var activation = "relu"

  def main(args: Array[String]): Unit = {
    var idx = -1

    idx = args.indexOf("--input")
    if (idx != -1)
      input = args(idx + 1)

    idx = args.indexOf("--windowSize")
    if (idx != -1)
      windowSize = args(idx + 1).toInt

    // Init spark
    val spark = SparkSession.builder()
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    // Load data
    val df = spark
      .read
      .option("header", "true")
      .csv(input)
      .select("Close", "Date")
      .toDF("value", "timestamp")

    val testDF = spark.read.option("header", "true")
      .csv(testInput).select("Close", "Date")
      .toDF("value", "timestamp")

    // Train
    val tsMLP = new TimeSeriesMLP()
      .setWindowSize(windowSize)
      .setHiddenLayers(hiddenLayers)
      .setActivation(activation)
      .setSeed(1234L)
      .setMaxIter(1000)
      .setBlockSize(32)

    val model = tsMLP.fit(df)

    // Test
    val testFeatures = TimeSeriesMLP.slidingWindowTransform(testDF, windowSize, sort = true)
    testFeatures.show()

    val predictUDF = spark.udf.register("prediction", (feature: Vector) => model.predict(feature))

    val predictions = testFeatures.withColumn("prediction", predictUDF(testFeatures("features")))
        .cache()

    val loss = predictions.select("label", "prediction").rdd.map { case  Row(label: Double, prediction: Double) =>
      math.pow(math.abs(label - prediction), 2)
    }.sum() / predictions.count()

    predictions.select("label", "prediction").write.option("header", "true").csv("src/test/resources/sigmoid.csv")

    predictions.show()

    println(s"activation = $activation, loss = $loss")
  }
}
