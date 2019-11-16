package org.apache.spark.ml.timeseries

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object TimeSeriesMLPTest {
  var input = "data/simple_ts.csv"
  var windowSize = 3
  var pattern = "dd-MM-yy"
  var hiddenLayers = Array(10)

  def main(args: Array[String]): Unit = {
    var idx = -1

    idx = args.indexOf("--input")
    if (idx != -1)
      input = args(idx + 1)

    idx = args.indexOf("--windowSize")
    if (idx != -1)
      windowSize = args(idx + 1).toInt

    val spark = SparkSession.builder()
      .master("local[*]")
      .getOrCreate()

    val df = spark
      .read
      .option("header", "true")
      .csv(input)
      .toDF("value", "timestamp")

    df.show()

    val tsMLP = new TimeSeriesMLP()
      .setWindowSize(windowSize)
      .setHiddenLayers(hiddenLayers)
      .setSeed(1234L)
      .setMaxIter(100)
      .setBlockSize(1)

    val model = tsMLP.fit(df)

    val testData = Array(
      Vectors.dense(1.0, 1.0, 1.0),
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.dense(10.0, 11.0, 12.0)
    )

    for (feature <- testData) {
      println(model.predict(feature))
    }
//    val predict = model.predict(Vectors.dense(1.0, 1.0, 1.0))
//    println(predict)

    spark.stop()
  }

}
