from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from DataManipulation import DataManipulation
from Estimators.XGBoost import XGBoost
from Transformers.FilterDepartment import FilterDepartment
from Transformers.ImputePrice import ImputePrice
from Transformers.LagFeature import LagFeature
from Transformers.LogTransformation import LogTransformation
from Transformers.MonthlyAggregate import MonthlyAggregate
from Transformers.NegativeSales import NegativeSales

import pandas as pd


def initialize_session(name):
    return SparkSession.builder.master("local").appName(name). \
        config("spark.driver.bindAddress", "localhost"). \
        config("spark.ui.port", "4050"). \
        getOrCreate()


if __name__ == '__main__':
    spark = initialize_session("Assignment")

    data = DataManipulation()
    df = data.get_data()

    # df = data.filter_store(df, "WI_1")
    filterDepartment = FilterDepartment(inputCol="FOODS_1", filterCol="dept_id")

    # filterStore = FilterStore()
    imputePrice = ImputePrice()
    negativeSales = NegativeSales(column="sales")
    aggregate = MonthlyAggregate(columns=["store_id", "dept_id", "year", "month"],
                                 expressions={"sales": "sum",
                                              "sell_price": "avg",
                                              "event_name_1": "count",
                                              "event_name_2": "count",
                                              "snap_WI": "sum"}
                                 )
    logTransformation = LogTransformation(inputCols=["sales"])
    lagFeatures = LagFeature(partitionBy=["store_id", "dept_id"],
                             orderBy=["year", "month"],
                             lags=[1, 3, 12],
                             target="sales"
                             )

    storeIndexer = StringIndexer(inputCol="store_id", outputCol="store_id_index")
    yearIndexer = StringIndexer(inputCol="year", outputCol="year_index")

    vector = VectorAssembler(inputCols=["store_id_index", "month", "year_index", "lag_1", "lag_3", "lag_12",
                                        "event_name_1", "event_name_2", "sell_price"],
                             outputCol="features")

    transformed = Pipeline(stages=[filterDepartment, imputePrice, negativeSales, aggregate,
                                   logTransformation, lagFeatures, storeIndexer,
                                   yearIndexer, vector]).fit(df).transform(df)

    print(transformed.columns)
    print(transformed.show(10))

    train, test = data.train_test_split(transformed)
    inputColumns = ["store_id_index", "month", "year_index", "lag_1",
                    "lag_3", "lag_12", "event_name_1", "event_name_2",
                    "sell_price"]
    xgbModel = XGBoost(inputCols=inputColumns, labelCol="sales").fit(train)
    pred = xgbModel.transform(test)
    p = pd.DataFrame([pred, test])
    print(p)
