package org.apache.spark.ml.timeseries

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.MLTest
import org.apache.spark.sql.Column
import org.apache.spark.sql.types.DoubleType

class TimeSeriesMLPSuite extends MLTest {

  case class TSData(value: Double, timestamp: String)

  test("TimeSeriesMLP") {
    val filePath = "data/simple_ts.csv"

    val df = spark
      .read
      .option("header", "true")
      .csv(filePath)
      .toDF("value", "timestamp")
      .withColumn("value", new Column("value").cast(DoubleType))

    val hiddenLayers = Array(10)
    val windowSize = 3

    val tsMLP = new TimeSeriesMLP()
      .setHiddenLayers(hiddenLayers)
      .setWindowSize(windowSize)
      .setPattern("yyyy-MM-dd")
      .setSeed(1234L)
      .setStepSize(0.001)
      .setMaxIter(10000)
      .setBlockSize(1)
      .setSolver("l-bfgs")

    val model = tsMLP.fit(df)

    val testData = Array(
      Vectors.dense(1.0, 1.0, 1.0),
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.dense(10.0, 11.0, 12.0)
    )

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
      .withColumn("value", new Column("value").cast(DoubleType))

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
      .withColumn("value", new Column("value").cast(DoubleType))

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
    )).toDF("value", "timestamp")

    model.setFutures(5)

    model.transform(testDF).show()
  }

  test("year") {

    val df = spark.createDataFrame(Seq(
      TSData(10.0, "1800-10-20"),
      TSData(11.0, "1900-10-20"),
      TSData(12.0, "2000-10-20"),
      TSData(13.0, "2100-10-20")
    )).toDF("value", "timestamp")

    val hiddenLayers = Array(10)
    val windowSize = 3

    val tsmlp = new TimeSeriesMLP()
      .setHiddenLayers(hiddenLayers)
      .setWindowSize(windowSize)
      .setPattern("yyyy-MM-dd")
      .setActivation("identity")
      .setSeed(1234L)
      .setStepSize(0.001)
      .setMaxIter(10000)
      .setUnit("year")
      .setFrequency(100)

    val model = tsmlp.fit(df)

    val predictions = model.setFutures(2).transform(df)

    predictions.show()
  }

  test("minute") {

    val df = spark.createDataFrame(Seq(
      TSData(10.0, "2012-12-31 23:00"),
      TSData(11.0, "2012-12-31 23:15"),
      TSData(12.0, "2012-12-31 23:30"),
      TSData(13.0, "2012-12-31 23:45"),
      TSData(14.0, "2013-1-1 00:00")
    )).toDF("value", "timestamp")

    val hiddenLayers = Array(10)
    val windowSize = 3

    val tsmlp = new TimeSeriesMLP()
      .setHiddenLayers(hiddenLayers)
      .setWindowSize(windowSize)
      .setPattern("yyyy-MM-dd HH:mm")
      .setActivation("identity")
      .setSeed(1234L)
      .setStepSize(0.001)
      .setMaxIter(10000)
      .setUnit("minute")
      .setFrequency(15)

    val model = tsmlp.fit(df)

    val predictions = model.setFutures(2).transform(df)

    predictions.show()
  }

}
