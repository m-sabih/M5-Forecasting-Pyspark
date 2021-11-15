from pyspark.sql import SparkSession

from Logging import Logging


class DataManipulation:
    log = Logging.getLogger()

    def __init__(self):
        self.spark = SparkSession.builder.getOrCreate()

    def read_data(self):
        calendarDf = self.spark.read.option("inferSchema", "true").option("header", "true").csv("./data/calendar.csv")
        modifiedSalesTrainDf = self.spark.read.option("inferSchema", "true").option("header", "true") \
            .csv("./data/modifiedsalesTrainDf.csv")
        sellPricesDf = self.spark.read.option("inferSchema", "true").option("header", "true") \
            .csv("./data/sell_prices.csv")
        return calendarDf, modifiedSalesTrainDf, sellPricesDf

    def get_data(self):
        DataManipulation.log.info("Reading data from files")
        calendarDf, modifiedSalesTrainDf, sellPricesDf = self.read_data()
        df = modifiedSalesTrainDf.join(calendarDf, modifiedSalesTrainDf.day == calendarDf.d, "left")
        df = df.drop("d")
        df = df.join(sellPricesDf, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        return df

    @staticmethod
    def filter_store(df, store_name):
        return df.filter(df.store_id == store_name)

    @classmethod
    def train_test_split(cls, df, year=2016):
        cls.log.info("Making train test splits")
        train, test = df[df['year'] < year], df[df['year'] >= year]
        return train, test
