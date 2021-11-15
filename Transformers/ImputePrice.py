from pyspark import keyword_only
from pyspark.ml import Transformer

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

        df_aggregated = df.groupBy(["id"]).avg("sell_price")
        df_aggregated = df_aggregated.withColumnRenamed("id", "agg_id")
        df = df.join(df_aggregated, df["id"] == df_aggregated["agg_id"], "inner")
        df = df.drop("agg_id", "sell_price")
        df = df.withColumnRenamed("avg(sell_price)", "sell_price")
        return df
