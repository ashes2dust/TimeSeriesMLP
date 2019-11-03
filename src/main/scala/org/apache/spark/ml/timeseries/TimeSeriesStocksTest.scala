package org.apache.spark.ml.timeseries

import org.apache.spark.sql.SparkSession

object TimeSeriesStocksTest {
  var input = "src/test/resources/sp500.csv"
  var windowSize = 10
  var pattern = "dd-MM-yy"
  var hiddenLayers = Array(100, 100)

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
      .select("Close", "Date")
      .toDF("value", "timestamp")

    val tsMLP = new TimeSeriesMLP()
      .setWindowSize(windowSize)
      .setHiddenLayers(hiddenLayers)
      .setSeed(1234L)
      .setMaxIter(100)
      .setBlockSize(32)

    val model = tsMLP.fit(df)
  }
}
