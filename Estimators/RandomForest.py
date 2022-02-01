from pyspark import keyword_only
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasFeaturesCol
from pyspark.ml import Estimator

from DataManipulation import DataManipulation
from Evaluator.MAPE import MAPE
from Logging import Logging
from pyspark.ml.regression import RandomForestRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from functools import partial


class RandomForest(Estimator, HasLabelCol, HasFeaturesCol, HasPredictionCol):
    searchSpace = {
        'maxDepth': hp.quniform('maxDepth', 1, 25, 1),
        'numTrees': hp.quniform('numTrees', 10, 1000, 5),
        'minInfoGain': hp.quniform('minInfoGain', 0.0, 0.7, 0.1),
        'subsamplingRate': hp.choice('subsamplingRate', [1, 0.9]),
        'maxBins': hp.quniform('maxBins', 40, 75, 1)
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

        bestParams = space_eval(self.searchSpace, best)
        print(bestParams)

        rf = RandomForestRegressor(featuresCol=featuresCol, labelCol=labelCol, **best)
        return rf.fit(train)
