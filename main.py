from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from DataManipulation import DataManipulation
from ImputePrice import ImputePrice
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
    # store = input()
    df = data.filter_store(df, "WI_1")

    imputeNegativePrice = ImputePrice()
    negativeSales = NegativeSales(column="sales")
    aggregate = MonthlyAggregate(columns=["store_id", "dept_id", "year", "month"],
                                 expressions={"sales": "sum",
                                              "sell_price": "avg",
                                              "event_name_1": "count",
                                              "event_name_2": "count",
                                              "snap_WI": "sum"}
                                 )
    transformed = Pipeline(stages=[imputeNegativePrice, negativeSales, aggregate]).fit(df).transform(df)

    print(transformed.show(5))
