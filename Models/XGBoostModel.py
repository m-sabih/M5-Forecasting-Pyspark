from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols

from Logging import Logging
import pandas as pd


class XGBoostModel(Model, HasLabelCol, HasInputCols, HasPredictionCol):
    model = Param(
        Params._dummy(),
        "model",
        "model for prediction",
        None,
    )

    @keyword_only
    def __init__(self, labelCol=None, inputCols=None, predictionCol=None, model=None):
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
        self.log.info("Making XGBoost model predictions")

        featureCols = self.getInputCols()
        labelCol = self.getLabelCol()
        pred = self.getPredictionCol()
        model = self.getModel()
        X = df[featureCols].toPandas()
        y = df.select(labelCol).toPandas()
        df = df.
        prediction = model.predict(X)
        p = pd.DataFrame({'prediction': prediction, 'actual': y[labelCol]})
        print(p)
        return p
