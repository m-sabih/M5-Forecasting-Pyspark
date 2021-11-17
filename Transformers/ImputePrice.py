from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.sql import Window
from pyspark.sql.functions import when, col, avg

from Logging import Logging


class ImputePrice(Transformer):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        self.log = Logging.getLogger()
        super().__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        self.log.info("Filling null prices with mean")

        w = Window.partitionBy(df.id)
        df = df.withColumn('sell_price', when(col('sell_price').isNull(),
                                              avg(col('sell_price')).over(w))
                           .otherwise(col('sell_price')))
        return df
