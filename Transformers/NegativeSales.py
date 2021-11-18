from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import functions as F

from Logging import Logging


class NegativeSales(Transformer):
    column = Param(
        Params._dummy(),
        "column",
        "Column remove negative sales from",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, column=None):
        self.log = Logging.getLogger()
        super().__init__()
        self._setDefault(column=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, column=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getColumn(self):
        return self.getOrDefault(self.column)

    def _transform(self, df):
        self.log.info("Filling negative sales with zero")

        column = self.getColumn()
        return df.withColumn(column, F.when(df[column] < 0, 0).when(F.col(column).isNull(), 0)
                             .otherwise(F.col(column)))
