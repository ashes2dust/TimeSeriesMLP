package org.apache.spark.ml.timeseries

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class MLPSuite extends FunSuite with BeforeAndAfterAll {
  var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName("TimeseriesWithMLPTest")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
  }


  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  test("Regression") {
    val filePath = this.getClass.getResource("/simple_ts.csv").toString

    val df = spark
      .read
      .option("header", "true")
      .csv(filePath)
      .toDF("value", "timestamp")

    val layers = Array(3, 10, 1)

    val tsMLP = new TimeSeriesMLP()
      .setLayers(layers)
      .setSeed(1234L)
      .setStepSize(0.001) // stepSize如果设置不当，应该是出现震荡的情况，无法收敛
      .setMaxIter(10000)
      .setBlockSize(1)
      .setSolver("gd")

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
