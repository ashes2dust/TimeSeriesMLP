package org.apache.spark.ml.regression

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.MLTest
import org.apache.spark.sql.DataFrame


class MLPRegressorSuite extends MLTest {

  case class TestData(features: Vector, label: Double)

  var trainDF: DataFrame = _
  var testDF: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    trainDF = spark.createDataFrame(Seq(
      TestData(Vectors.dense(1.0, 2.0, 3.0), 4.0),
      TestData(Vectors.dense(2.0, 3.0, 4.0), 5.0),
      TestData(Vectors.dense(3.0, 4.0, 5.0), 6.0)
    ))

    testDF = spark.createDataFrame(Seq(
      TestData(Vectors.dense(4.0, 5.0, 6.0), 7.0),
      TestData(Vectors.dense(5.0, 6.0, 7.0), 8.0),
      TestData(Vectors.dense(6.0, 7.0, 8.0), 9.0)
    ))
  }

  test("MLPRegression") {

    val testFeature = Vectors.dense(8.0, 9.0, 10.0)

    val layers = Array(3, 10, 1)

    val mlpRegressor = new MultilayerPerceptronRegressor()
      .setLayers(layers)

    val model = mlpRegressor.fit(trainDF)

    println(model.predict(testFeature))

    model.predict(testDF).show(20)
  }

  test("sigmoid activation") {
    val layers = Array(3, 10, 1)

    val mlpRegressor = new MultilayerPerceptronRegressor()
      .setLayers(layers)
      .setActivation("sigmoid")

    val model = mlpRegressor.fit(trainDF)

    model.predict(testDF).show(20)
  }

  test("identity activation") {
    val layers = Array(3, 10, 1)

    val mlpRegressor = new MultilayerPerceptronRegressor()
      .setLayers(layers)
      .setActivation("identity")

    val model = mlpRegressor.fit(trainDF)

    model.predict(testDF).show(20)
  }

  test("tanh activation") {
    val layers = Array(3, 10, 1)

    val mlpRegressor = new MultilayerPerceptronRegressor()
      .setLayers(layers)
      .setActivation("tanh")

    val model = mlpRegressor.fit(trainDF)

    model.predict(testDF).show(20)
  }

  test("LRRegression") {

    val trainDF = spark.createDataFrame(Seq(
      TestData(Vectors.dense(1.0, 2.0, 3.0), 4.0),
      TestData(Vectors.dense(2.0, 3.0, 4.0), 5.0),
      TestData(Vectors.dense(3.0, 4.0, 5.0), 6.0)
    ))

    val testFeature = Vectors.dense(3.0, 4.0, 5.0)

    val regressor = new LinearRegression()

    val model = regressor.fit(trainDF)

    println(model.predict(testFeature))
  }
}
