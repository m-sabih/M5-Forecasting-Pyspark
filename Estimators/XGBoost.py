from pyspark import keyword_only
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols
from pyspark.ml import Estimator
import xgboost as xgb

from Models.XGBoostModel import XGBoostModel


class XGBoost(Estimator, HasLabelCol, HasInputCols, HasPredictionCol):

    @keyword_only
    def __init__(self, labelCol=None, inputCols=None, predictionCol=None):
        super().__init__()
        self._setDefault(predictionCol="prediction")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, inputCols=None, predictionCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, df):
        featuresCol = self.getInputCols()
        labelCol = self.getLabelCol()
        predCol = self.getPredictionCol()
        X = df[featuresCol].toPandas()
        y = df.select(labelCol).toPandas()
        xgboost = xgb.XGBRegressor().fit(X, y)
        return XGBoostModel(xgboost, labelCol, featuresCol, predCol)
