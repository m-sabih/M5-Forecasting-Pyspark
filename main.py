from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from DataManipulation import DataManipulation
from Estimators.ProphetEstimator import ProphetEstimator
from Estimators.RandomForest import RandomForest
from Estimators.XGBoost import XGBoost
from Logging import Logging
from Transformers.FilterDepartment import FilterDepartment
from Transformers.ImputePrice import ImputePrice
from Transformers.LagFeature import LagFeature
from Transformers.LogTransformation import LogTransformation
from Transformers.MonthlyAggregate import MonthlyAggregate
from Transformers.NegativeSales import NegativeSales
import findspark
from functools import reduce
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from Transformers.Scaling import Scaling


def initialize_session(name):
    return SparkSession.builder.master("local[*]").appName(name). \
        config("spark.driver.bindAddress", "localhost"). \
        config("spark.ui.port", "4050"). \
        getOrCreate()


def union_all(*dfs):
    return reduce(DataFrame.union, dfs)


if __name__ == '__main__':
    findspark.init()
    spark = initialize_session("Assignment")
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    log = Logging.getLogger()
    log.info("Initializing session")

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
                             lags=[i for i in range(1, 12)],
                             target="sales"
                             )

    storeIndexer = StringIndexer(inputCol="store_id", outputCol="store_id_index")
    yearIndexer = StringIndexer(inputCol="year", outputCol="year_index")
    scaling = Scaling()

    inputColumns = ["store_id_index", "month", "year_index", "event_name_1", "event_name_2", "sell_price"]
    inputColumns.extend(["lag_{}".format(i) for i in range(1, 12)])
    vector = VectorAssembler(inputCols=inputColumns,
                             outputCol="features")

    log.info("Initiating pipeline")
    transformed = Pipeline(stages=[filterDepartment, imputePrice, negativeSales, aggregate,
                                   logTransformation, lagFeatures, storeIndexer,
                                   yearIndexer, scaling, vector]).fit(df).transform(df)

    log.info("Preprocessing pipeline completed")

    train, test = data.train_test_split(transformed)

    xgbModel = XGBoost(inputCols=inputColumns, labelCol="sales").fit(train)
    predXgb = xgbModel.transform(test)
    print(predXgb.show(10))

    rfModel = RandomForest(inputCols=inputColumns, labelCol="sales").fit(train)
    predRf = rfModel.transform(test)
    print(predRf.show(10))

    transformedProphet = transformed.withColumn("ds", F.to_date(F.concat_ws("-", "Year", "month")))
    trainProphet, testProphet = data.train_test_split(transformedProphet)


    def getStores(data):
        storesName = data.select("store_id").distinct().collect()
        stores = {}
        for store in storesName:
            stores[store.store_id] = data.filter(df["store_id"] == store.store_id)
        return stores


    trainStores = getStores(trainProphet)
    testStores = getStores(testProphet)

    models = {}
    for key, value in trainStores.items():
        prophetModel = ProphetEstimator(labelCol="sales").fit(value)
        models[key] = prophetModel

    predictions = {}
    for key, value in models.items():
        data = testStores[key]
        pred = value.transform(data)
        predictions[key] = pred

    predProphet = union_all(*predictions.values())

    print(predProphet.show(10))
