from pyspark import keyword_only
from pyspark.ml.param.shared import HasLabelCol, HasFeaturesCol, HasPredictionCol
from pyspark.ml import Estimator
import xgboost as xgb

from Models.XGBoostModel import XGBoostModel


class XGBoost(Estimator, HasLabelCol, HasFeaturesCol, HasPredictionCol):

    @keyword_only
    def __init__(self, labelCol=None, featuresCol=None, predictionCol=None):
        super().__init__()
        self._setDefault(predictionCol="prediction")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, featuresCol=None, predictionCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, df):
        featuresCol = self.getFeaturesCol()
        labelCol = self.getLabelCol()
        predCol = self.getPredictionCol()
        X = df[featuresCol]
        y = df[labelCol]
        xgboost = xgb.XGBRegressor().fit(X, y)
        return XGBoostModel(xgboost, labelCol, featuresCol, predCol)
