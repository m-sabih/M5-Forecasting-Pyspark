import findspark
import numpy
import pytest
import pandas as pd
from pyspark.sql import SparkSession
from Evaluator.MAPE import MAPE
from Transformers.FilterDepartment import FilterDepartment


@pytest.fixture(scope="session")
def spark():
    findspark.init()
    spark = SparkSession.builder.appName("testing-session").getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    return spark


def test_mape_evaluation_score(spark):
    dept = FilterDepartment(inputCol="Food", filterCol="Department")
    df = pd.DataFrame(data=[[3, 2.5], [-0.5, 0.0], [2, 2], [7, 8]], columns=["actual", "prediction"])
    df = spark.createDataFrame(df)
    df.createOrReplaceTempView('df')

    mape = MAPE(labelCol="actual", predictionCol="prediction")
    score = mape.evaluate(df)

    print(score)
    assert score == 0.3273809523809524
