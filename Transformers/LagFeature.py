from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.sql import Window
from pyspark.sql.functions import lag

from Logging import Logging


class LagFeature(Transformer):
    partitionBy = Param(
        Params._dummy(),
        "partitionBy",
        "Columns used to create partition window",
        typeConverter=TypeConverters.toListString,
    )

    orderBy = Param(
        Params._dummy(),
        "orderBy",
        "Columns to group partition window on",
        typeConverter=TypeConverters.toListString,
    )

    lags = Param(
        Params._dummy(),
        "lags",
        "ranges for lags",
        typeConverter=TypeConverters.toListInt,
    )

    target = Param(
        Params._dummy(),
        "target",
        "target variable on which lags needs to be added",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, partitionBy=None, orderBy=None,
                 lags=None, target=None):
        self.log = Logging.getLogger()
        super().__init__()
        self._setDefault(partitionBy=None, orderBy=None, lags=None, target=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, partitionBy=None, orderBy=None,
                  lags=None, target=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getPartitionBy(self):
        return self.getOrDefault(self.partitionBy)

    def getOrderBy(self):
        return self.getOrDefault(self.orderBy)

    def getLags(self):
        return self.getOrDefault(self.lags)

    def getTarget(self):
        return self.getOrDefault(self.target)

    def _transform(self, df):
        self.log.info("Adding lags")

        partitionBy = self.getPartitionBy()
        orderBy = self.getOrderBy()
        lags = self.getLags()
        target = self.getTarget()

        windowSpec = Window.partitionBy(partitionBy).orderBy(orderBy)
        for lag_val in lags:
            df = df.withColumn('lag_' + str(lag_val), lag(target, lag_val).over(windowSpec))

        for lag_val in lags:
            name = "lag_{}".format(lag_val)
            df = df.fillna({name: '0'})

        return df
