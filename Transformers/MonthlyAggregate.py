from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters


class MonthlyAggregate(Transformer):
    columns = Param(
        Params._dummy(),
        "columns",
        "Columns to group by on",
        typeConverter=TypeConverters.toListString,
    )

    expressions = Param(
        Params._dummy(),
        "expressions",
        "Dictionary of aggregate expressions",
        None,
    )

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, columns=None, expressions=None):
        super().__init__()
        self._setDefault(columns=None, expressions=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def getColumns(self):
        return self.getOrDefault(self.columns)

    def getExpression(self):
        return self.getOrDefault(self.expressions)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, columns=None, expressions=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, df):
        groupByColumns = self.getColumns()
        aggregateExpression = dict(self.getExpression())
        df_agg = df.groupBy(groupByColumns).agg(aggregateExpression)
        for key, opr in aggregateExpression.items():
            name = "{}({})".format(opr, key)
            df_agg = df_agg.withColumnRenamed(name, key)
        return df_agg
