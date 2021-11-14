from pyspark import keyword_only
from pyspark.ml import Model
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasLabelCol, HasFeaturesCol, HasPredictionCol


class XGBoostModel(Model, HasLabelCol, HasFeaturesCol, HasPredictionCol):

    model = Param(
        Params._dummy(),
        "model",
        "model for prediction",
        None,
    )

    @keyword_only
    def __init__(self, labelCol=None, inputCols=None, predictionCol=None, model=None):
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
        featureCols = self.getFeaturesCol()
        pred = self.getPredictionCol()
        model = self.getModel()
        X = df[featureCols].toPandas()
        prediction = model.predict(X)
        return prediction