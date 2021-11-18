from pyspark import keyword_only
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol
from pyspark.sql import DataFrame
from pyspark.sql.functions import abs, mean


class MAPE(Evaluator, HasLabelCol, HasPredictionCol):

    @keyword_only
    def __init__(self, labelCol=None, predictionCol=None):
        super().__init__()
        self._setDefault(labelCol=None, predictionCol=None)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, labelCol=None, predictionCol=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getLabelCol(self):
        return self.getOrDefault(self.labelCol)

    def getPredictionCol(self):
        return self.getOrDefault(self.predictionCol)

    def _evaluate(self, df: DataFrame):
        labelCol = self.getLabelCol()
        predictionCol = self.getPredictionCol()
        score = df.select(mean(abs((df[labelCol] - df[predictionCol]) / df[labelCol])))
        return score.first()[0]
