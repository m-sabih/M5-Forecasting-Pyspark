from pyspark import keyword_only
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols
from pyspark.ml import Estimator
import xgboost as xgb
from Logging import Logging
from Models.XGBoostModel import XGBoostModel


class XGBoost(Estimator, HasLabelCol, HasInputCols, HasPredictionCol):

    @keyword_only
    def __init__(self, labelCol=None, inputCols=None, predictionCol=None):
        self.log = Logging.getLogger()
        super().__init__()
        self._setDefault(predictionCol="prediction")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, inputCols=None, predictionCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, df):
        self.log.info("Training XGBoost")
        featuresCol = self.getInputCols()
        labelCol = self.getLabelCol()
        predCol = self.getPredictionCol()
        X = df[featuresCol].toPandas()
        y = df.select(labelCol).toPandas()
        xgboost = xgb.XGBRegressor().fit(X, y[labelCol])
        return XGBoostModel(labelCol=labelCol, inputCols=featuresCol, predictionCol=predCol, model=xgboost)
