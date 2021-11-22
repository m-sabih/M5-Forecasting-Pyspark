from functools import partial

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
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
        labels = self.getLabelCol()
        predCol = self.getPredictionCol()

        if params is None:
            prophet = Prophet()
        else:
            prophet = Prophet(**params)

        prophetModel = prophet.fit(train)
        predictions = ProphetModel(labelCol=labels, predictionCol=predCol, model=prophetModel) \
            .transform(validation)

        mape = MAPE(labelCol="sales", predictionCol=predCol)
        score = mape.evaluate(predictions)
        print("score:", score)
        return {'loss': score, 'status': STATUS_OK, 'model': prophetModel}

    def _fit(self, df):
        self.log.info("Training XGBoost")
        print("Training XGBoost")
        labelCol = self.getLabelCol()
        predCol = self.getPredictionCol()

        df = df.withColumnRenamed(labelCol, "y")
        data = DataManipulation()
        train, validation = data.train_test_split(df, 2015)

        train = train.select("ds", "y")
        train = train.toPandas()

        self.trainModel(train, validation, None)

        trials = Trials()
        best = fmin(partial(self.trainModel, train, validation),
                    space=ProphetEstimator.searchSpace,
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials)
        print(best)

        X = df.toPandas()
        prophetBest = Prophet(**best)
        prophetBest = prophetBest.fit(X)
        return ProphetModel(labelCol=labelCol, predictionCol=predCol, model=prophetBest)

