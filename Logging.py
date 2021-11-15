from pyspark.sql import SparkSession


class Logging:
    spark = SparkSession.builder.getOrCreate()

    @classmethod
    def getLogger(cls):
        cls.spark.sparkContext.setLogLevel('INFO')
        log4jLogger = cls.spark._jvm.org.apache.log4j
        return log4jLogger.LogManager.getLogger(__name__)