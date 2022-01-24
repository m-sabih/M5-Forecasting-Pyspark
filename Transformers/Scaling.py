from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols
from pyspark.sql.functions import col, mean, stddev
from Logging import Logging


class Scaling(Transformer, HasInputCols):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        self.log = Logging.getLogger()
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        self.log.info("Scaling")
        columns = self.getInputCols()
        for column in columns:
            meanCol, sttdevCol = df.select(mean(column), stddev(column)).first()
            df = df.withColumn(column+"_scaled", (col(column) - meanCol) / sttdevCol)

        return df
