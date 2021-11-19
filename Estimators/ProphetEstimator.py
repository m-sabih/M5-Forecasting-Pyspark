from pyspark import keyword_only
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols
from pyspark.ml import Estimator

from Logging import Logging
from Models.ProphetModel import ProphetModel
from fbprophet import Prophet


class ProphetEstimator(Estimator, HasLabelCol, HasInputCols, HasPredictionCol):
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
        self.log.info("Training Prophet")
        print("Training Prophet")
        label = self.getLabelCol()
        predCol = self.getPredictionCol()

        df = df.withColumnRenamed(label, "y")
        df = df.select("ds", "y")
        X = df.toPandas()

        model = Prophet()
        prophet = model.fit(X)

        return ProphetModel(labelCol=label, predictionCol=predCol, model=prophet)
