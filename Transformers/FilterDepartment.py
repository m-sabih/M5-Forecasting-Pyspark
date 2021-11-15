from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.param.shared import HasInputCol

from Logging import Logging


class FilterDepartment(Transformer, HasInputCol):
    filterCol = Param(
        Params._dummy(),
        "filterCol",
        "Filter departments",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(self, inputCol=None, filterCol=None):
        self.log = Logging.getLogger()
        super().__init__()
        self._setDefault(filterCol=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, filterCol=None):
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def getFilterCol(self):
        return self.getOrDefault(self.filterCol)

    def _transform(self, df):
        self.log.info("Filtering departments")
        input_col = self.getInputCol()
        filter_col = self.getFilterCol()
        return df.filter(df[filter_col] == input_col)
