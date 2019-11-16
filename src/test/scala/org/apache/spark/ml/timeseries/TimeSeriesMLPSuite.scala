package org.apache.spark.ml.timeseries

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.MLTest

class TimeSeriesMLPSuite extends MLTest {

  case class TSData(value: Double, timestamp: String)

  test("TimeSeriesMLP") {
    val filePath = "data/simple_ts.csv"

    val df = spark
      .read
      .option("header", "true")
      .csv(filePath)
      .toDF("value", "timestamp")

    val hiddenLayers = Array(10)
    val windowSize = 3

    val tsMLP = new TimeSeriesMLP()
      .setHiddenLayers(hiddenLayers)
      .setWindowSize(windowSize)
      .setPattern("yyyy-MM-dd")
      .setSeed(1234L)
      .setStepSize(0.001) // stepSize如果设置不当，应该是出现震荡的情况，无法收敛
      .setMaxIter(10000)
      .setBlockSize(1)
      .setSolver("l-bfgs")

    val model = tsMLP.fit(df)

    val testData = Array(
      Vectors.dense(1.0, 1.0, 1.0),
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.dense(10.0, 11.0, 12.0)
    )
    println(s"model size: ${model.layers.length}")

    for (feature <- testData) {
      println(model.predict(feature))
    }

  }

  test("TimeSeriesMLP with identity") {
    val filePath = "data/simple_ts.csv"

    val df = spark
      .read
      .option("header", "true")
      .csv(filePath)
      .toDF("value", "timestamp")

    val hiddenLayers = Array(10)
    val windowSize = 3

    val tsMLP = new TimeSeriesMLP()
      .setHiddenLayers(hiddenLayers)
      .setWindowSize(windowSize)
      .setPattern("yyyy-MM-dd")
      .setActivation("identity")
      .setSeed(1234L)
      .setStepSize(0.001)
      .setMaxIter(10000)
      .setSolver("l-bfgs")

    val model = tsMLP.fit(df)

    val testData = Array(
      Vectors.dense(1.0, 1.0, 1.0),
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.dense(10.0, 11.0, 12.0)
    )
    println(s"model size: ${model.layers.length}")

    for (feature <- testData) {
      println(model.predict(feature))
    }

  }

  test("predict dataframe") {
    val filePath = "data/simple_ts.csv"

    val df = spark
      .read
      .option("header", "true")
      .csv(filePath)
      .toDF("value", "timestamp")

    val hiddenLayers = Array(10)
    val windowSize = 3

    val tsMLP = new TimeSeriesMLP()
      .setHiddenLayers(hiddenLayers)
      .setWindowSize(windowSize)
      .setPattern("yyyy-MM-dd")
      .setActivation("identity")
      .setSeed(1234L)
      .setStepSize(0.001)
      .setMaxIter(10000)
      .setSolver("l-bfgs")

    val model = tsMLP.fit(df)

    val testDF = spark.createDataFrame(Seq(
      TSData(10.0, "2019-10-20"),
      TSData(11.0, "2019-10-21"),
      TSData(12.0, "2019-10-22"),
      TSData(13.0, "2019-10-23")
    ))

    model.setPredictDays(5)

    model.predict(testDF).show()
  }

}
