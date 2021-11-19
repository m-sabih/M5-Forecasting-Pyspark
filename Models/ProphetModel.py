from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols
from pyspark.sql import SparkSession

from Logging import Logging
import pandas as pd
import findspark


class ProphetModel(Model, HasLabelCol, HasInputCols, HasPredictionCol):
    findspark.init()
    model = Param(
        Params._dummy(),
        "model",
        "model for prediction",
        None,
    )

    @keyword_only
    def __init__(self, labelCol=None, inputCols=None, predictionCol=None, model=None):
        self.spark = SparkSession.builder.getOrCreate()
        self.log = Logging.getLogger()
        super().__init__()
        self._setDefault(model=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, inputCols=None, predictionCol=None, model=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getModel(self):
        return self.getOrDefault(self.model)

    def _transform(self, df):
        self.log.info("Making Prophet model predictions")

        labelCol = self.getLabelCol()
        pred = self.getPredictionCol()
        model = self.getModel()

        df = df.withColumnRenamed(labelCol, "y")
        X = df.toPandas()
        X_test = X[["ds"]]

        prediction = model.predict(X_test)
        prediction = prediction[["yhat"]]
        resultDf = pd.DataFrame({"store": X["store_id"], "year": X["year"], "month": X["month"],
                                 'sales': X["y"]})
        resultDf = pd.concat([resultDf, prediction], axis=1)
        result = self.spark.createDataFrame(resultDf)
        result.createOrReplaceTempView('result')
        return result
