import findspark
import numpy

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan

from Transformers.FilterDepartment import FilterDepartment
from Transformers.ImputePrice import ImputePrice
from Transformers.LagFeature import LagFeature
from Transformers.LogTransformation import LogTransformation
from Transformers.MonthlyAggregate import MonthlyAggregate
from Transformers.NegativeSales import NegativeSales
from Transformers.Scaling import Scaling


@pytest.fixture(scope="session")
def spark():
    findspark.init()
    spark = SparkSession.builder.appName("testing-session").getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return spark


def test_filter_department(spark):
    dept = FilterDepartment(inputCol="Food", filterCol="Department")
    df = pd.DataFrame(data=[["Food", "Apple"], ["Food", "Orange"], ["Game", "RDR2"]], columns=["Department", "Item"])
    df = spark.createDataFrame(df)
    df.createOrReplaceTempView('df')

    assert df.count() == 3
    df = dept.transform(df)
    assert df.count() == 2


def test_impute_price(spark):
    df = pd.DataFrame(data=[[1, 5], [1, numpy.nan], [1, 10], [2, 10], [2, numpy.nan]], columns=["id", "sell_price"])
    df = spark.createDataFrame(df)
    df.createOrReplaceTempView('df')

    assert df.where(col("sell_price").isNull() | isnan("sell_price")).count() == 2
    imputePrice = ImputePrice()
    df = imputePrice.transform(df)
    assert df.where(col("sell_price").isNull() | isnan("sell_price")).count() == 0


def test_negative_sales(spark):
    df = pd.DataFrame(data=[[1, -5], [1, 5], [2, -1]], columns=["id", "sales"])
    df = spark.createDataFrame(df)
    df.createOrReplaceTempView('df')

    assert df.filter(df["sales"] < 0).count() == 2
    negativeSales = NegativeSales(column="sales")
    df = negativeSales.transform(df)
    assert df.filter(df["sales"] < 0).count() == 0


def test_monthly_aggregation(spark):
    df = pd.DataFrame(data=[[1, 5], [1, 5], [2, 10], [2, 5]], columns=["id", "sales"])
    df = spark.createDataFrame(df)
    df.createOrReplaceTempView('df')

    aggregate = MonthlyAggregate(columns=["id"], expressions={"sales": "sum"})
    df = aggregate.transform(df)
    assert df.count() == 2
    assert df.collect()[0]["sales"] == 10
    assert df.collect()[1]["sales"] == 15

def test_log_transformation(spark):
    df = pd.DataFrame(data=[[1, 5]], columns=["id", "sales"])
    df = spark.createDataFrame(df)
    df.createOrReplaceTempView('df')

    logTransformation = LogTransformation(inputCols=["sales"])
    df = logTransformation.transform(df)
    print(df.show())
    assert df.collect()[0]["sales"] == 0.6989700043360189

def test_lags(spark):
    df = pd.DataFrame(data=[[1, 1, 1, 5],
                            [1, 1, 2, 10]], columns=["id", "dept_id", "year", "sales"])
    df = spark.createDataFrame(df)
    df.createOrReplaceTempView('df')

    lagFeatures = LagFeature(partitionBy=["id", "dept_id"],
                             orderBy=["year"],
                             lags=[1],
                             target="sales"
                             )

    df = lagFeatures.transform(df)
    assert df.collect()[0]["lag_1"] == 0
    assert df.collect()[1]["lag_1"] == 5

def test_scaling(spark):
    df = pd.DataFrame(data=[[1, 1],
                            [1, 2],
                            [1, 3],
                            [1, 4],
                            [1, 5]], columns=["id", "sell_price"])
    df = spark.createDataFrame(df)
    df.createOrReplaceTempView('df')

    scaling = Scaling(inputCols=["sell_price"])
    df = scaling.transform(df)
    print(df.show())
    assert df.collect()[2]["sell_price_scaled"] == 0
    assert abs(df.collect()[1]["sell_price_scaled"]) == abs(df.collect()[3]["sell_price_scaled"])
    assert abs(df.collect()[0]["sell_price_scaled"]) == abs(df.collect()[4]["sell_price_scaled"])
