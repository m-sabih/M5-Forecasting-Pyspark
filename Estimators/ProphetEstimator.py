from functools import partial

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from pyspark import keyword_only
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols
from pyspark.ml import Estimator

from DataManipulation import DataManipulation
from Evaluator.MAPE import MAPE
from Logging import Logging
from Models.ProphetModel import ProphetModel
from fbprophet import Prophet


class ProphetEstimator(Estimator, HasLabelCol, HasInputCols, HasPredictionCol):
    searchSpace = {
        'changepoint_prior_scale': hp.choice('changepoint_prior_scale', np.arange(0.2, 0.7, 0.1, dtype=float)),
        'holidays_prior_scale': hp.choice('holidays_prior_scale', np.arange(0.1, 0.7, 0.1, dtype=float)),
        'n_changepoints': hp.choice('n_changepoints', np.arange(10, 50, 5, dtype=int)),
    }

    @keyword_only
    def __init__(self, labelCol=None, inputCols=None, predictionCol=None):
        self.log = Logging.getLogger()
        super().__init__()
        self._setDefault(predictionCol="prediction")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, labelCol=None, inputCols=None, predictionCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def trainModel(self, train, validation, params):
        if params is None:
            prophet = Prophet()
        else:
            prophet = Prophet(**params)

        prophetModel = prophet.fit(train, iter=3000)
        #predictions = ProphetModel(labelCol=labels, predictionCol=predCol, model=prophetModel) \
        #    .transform(validation)
        predictions = prophetModel.predict(validation)

        #mape = MAPE(labelCol="sales", predictionCol=predCol)
        #score = mape.evaluate(predictions)

        score = np.mean(np.abs((validation['y'] - predictions['yhat']) / validation['y']))
        print("score:", score)
        return {'loss': score, 'status': STATUS_OK, 'model': prophetModel}

    def _fit(self, df):
        self.log.info("Training Prophet")
        print("Training Prophet")
        labelCol = self.getLabelCol()
        predCol = self.getPredictionCol()

        df = df.withColumnRenamed(labelCol, "y")
        data = DataManipulation()
        train, validation = data.train_test_split(df, 2015)

        train = train.select("ds", "y")
        train = train.toPandas()
        validation = validation.toPandas()

        self.trainModel(train, validation, None)

        trials = Trials()
        best = fmin(partial(self.trainModel, train, validation),
                    space=ProphetEstimator.searchSpace,
                    algo=tpe.suggest,
                    max_evals=5,
                    trials=trials)
        bestParams = space_eval(self.searchSpace, best)
        print(bestParams)

        X = df.toPandas()
        prophetBest = Prophet(**bestParams)
        prophetBest = prophetBest.fit(X)
        return ProphetModel(labelCol=labelCol, predictionCol=predCol, model=prophetBest)

