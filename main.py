from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from DataManipulation import DataManipulation
from FilterDepartment import FilterDepartment
from ImputePrice import ImputePrice
from LagFeature import LagFeature
from LogTransformation import LogTransformation
from MonthlyAggregate import MonthlyAggregate
from NegativeSales import NegativeSales


def initialize_session(name):
    return SparkSession.builder.master("local").appName(name). \
        config("spark.driver.bindAddress", "localhost"). \
        config("spark.ui.port", "4050"). \
        getOrCreate()


if __name__ == '__main__':
    spark = initialize_session("Assignment")

    data = DataManipulation()
    df = data.get_data()

    #df = data.filter_store(df, "WI_1")
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

    transformed = Pipeline(stages=[filterDepartment, imputePrice, negativeSales, aggregate,
                                   logTransformation, lagFeatures]).fit(df).transform(df)

    print(transformed.show(5))
