from pyspark import keyword_only
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasFeaturesCol
from pyspark.ml import Estimator

from DataManipulation import DataManipulation
from Evaluator.MAPE import MAPE
from Logging import Logging
from pyspark.ml.regression import RandomForestRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial
import numpy as np


class RandomForest(Estimator, HasLabelCol, HasFeaturesCol, HasPredictionCol):
    searchSpace = {
        'maxDepth': hp.choice('maxDepth', np.arange(1, 15, 1, dtype=int)),
        'maxBins': hp.choice('maxBins', np.arange(30, 70, 2, dtype=int)),
        'numTrees': hp.choice('numTrees', range(10, 1000, 5)),
        'minInfoGain': hp.choice('minInfoGain', [0, 0.1, 0.3, 0.7]),
        'subsamplingRate': hp.choice('subsamplingRate', [1, 0.9])
    }

    @keyword_only
    def __init__(self, labelCol=None, featuresCol=None, predictionCol=None):
        self.log = Logging.getLogger()
        super().__init__()
        self._setDefault(predictionCol="prediction")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, featuresCol=None, predictionCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def trainModel(self, train, validation, params):
        features = self.getFeaturesCol()
        labels = self.getLabelCol()

        if params is None:
            rf = RandomForestRegressor(featuresCol=features, labelCol=labels)
        else:
            rf = RandomForestRegressor(featuresCol=features, labelCol=labels, **params)

        rf = rf.fit(train)
        predictions = rf.transform(validation)
        mape = MAPE(labelCol="sales", predictionCol=self.getPredictionCol())
        score = mape.evaluate(predictions)
        print("score:", score)
        return {'loss': score, 'status': STATUS_OK, 'model': rf}

    def _fit(self, df):
        self.log.info("Training Random Forest")
        print("Training Random Forest")
        featuresCol = self.getFeaturesCol()
        labelCol = self.getLabelCol()

        data = DataManipulation()
        train, validation = data.train_test_split(df, 2015)

        self.trainModel(train, validation, None)

        trials = Trials()
        best = fmin(partial(self.trainModel, train, validation),
                    space=RandomForest.searchSpace,
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials)
        print(best)

        rf = RandomForestRegressor(featuresCol=featuresCol, labelCol=labelCol, **best)
        return rf.fit(train)
