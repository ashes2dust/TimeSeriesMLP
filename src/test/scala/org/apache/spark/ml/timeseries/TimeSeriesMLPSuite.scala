package org.apache.spark.ml.timeseries

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.MLTest

class TimeSeriesMLPSuite extends MLTest {

  test("TimeSeriesMLP") {
    val filePath = "src/test/resources/simple_ts.csv"

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
}
