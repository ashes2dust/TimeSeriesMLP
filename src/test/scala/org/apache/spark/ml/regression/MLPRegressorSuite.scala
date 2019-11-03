package org.apache.spark.ml.regression

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.MLTest


class MLPRegressorSuite extends MLTest {

  case class TestData(features: Vector, label: Double)

  test("MLPRegression") {

    val trainDF = spark.createDataFrame(Seq(
      TestData(Vectors.dense(1.0, 2.0, 3.0), 4.0),
      TestData(Vectors.dense(2.0, 3.0, 4.0), 5.0),
      TestData(Vectors.dense(3.0, 4.0, 5.0), 6.0)
    ))

    val testFeature = Vectors.dense(3.0, 4.0, 5.0)

    val layers = Array(3, 10, 1)

    val mlpRegressor = new MultilayerPerceptronRegressor()
      .setLayers(layers)

    val model = mlpRegressor.fit(trainDF)

    println(model.predict(testFeature))
  }

  test("LRRegression") {

    val trainDF = spark.createDataFrame(Seq(
      TestData(Vectors.dense(1.0, 2.0, 3.0), 4.0),
      TestData(Vectors.dense(2.0, 3.0, 4.0), 5.0),
      TestData(Vectors.dense(3.0, 4.0, 5.0), 6.0)
    ))

    val testFeature = Vectors.dense(3.0, 4.0, 5.0)

    val layers = Array(3, 10, 1)

    val regressor = new LinearRegression()

    val model = regressor.fit(trainDF)

    println(model.predict(testFeature))
  }
}
