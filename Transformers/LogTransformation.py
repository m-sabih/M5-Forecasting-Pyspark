from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols
from pyspark.sql.functions import log10, col


class LogTransformation(Transformer, HasInputCols):

    @keyword_only
    def __init__(self, inputCols=None):
        super().__init__()
        self._setDefault()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _transform(self, df):
        inputCols = self.getInputCols()
        for inputCol in inputCols:
            df = df.withColumn(inputCol, log10(col(inputCol)))
        return df