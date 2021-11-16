from pyspark import keyword_only
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasInputCols
from pyspark.ml import Estimator
import xgboost as xgb

from DataManipulation import DataManipulation
from Evaluator.MAPE import MAPE
from Logging import Logging
from Models.XGBoostModel import XGBoostModel
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial
import numpy as np


class XGBoost(Estimator, HasLabelCol, HasInputCols, HasPredictionCol):
    searchSpace = {
        'max_depth': hp.choice('max_depth', np.arange(10, 25, 1, dtype=int)),
        'n_estimators': hp.choice('n_estimators', np.arange(10, 1000, 10, dtype=int)),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1.0, 0.1),
        'min_child_weight': hp.choice('min_child_weight', np.arange(250, 350, 10, dtype=int)),
        'subsample': hp.quniform('subsample', 0.7, 0.9, 0.1),
        'eta': hp.quniform('eta', 0.1, 0.3, 0.1),
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

    def trainModel(self, X_train, y_train, validation, params):
        features = self.getInputCols()
        labels = self.getLabelCol()

        xgboost = xgb.XGBRegressor(**params)

        xgboost = xgboost.fit(X_train, y_train)
        predictions = XGBoostModel(labelCol=labels, inputCols=features,
                                   predictionCol=self.getPredictionCol(), model=xgboost) \
            .transform(validation)
        mape = MAPE(labelCol="actual", predictionCol=self.getPredictionCol())
        score = mape.evaluate(predictions)
        print("score:", score)
        return {'loss': score, 'status': STATUS_OK, 'model': xgboost}

    def _fit(self, df):
        self.log.info("Training XGBoost")
        print("Training XGBoost")
        featuresCol = self.getInputCols()
        labelCol = self.getLabelCol()
        predCol = self.getPredictionCol()

        data = DataManipulation()
        train, validation = data.train_test_split(df, 2015)

        X_train = train[featuresCol].toPandas()
        y_train = train.select(labelCol).toPandas()

        trials = Trials()
        best = fmin(partial(self.trainModel, X_train, y_train, validation),
                    space=XGBoost.searchSpace,
                    algo=tpe.suggest,
                    max_evals=10,
                    trials=trials)
        print(best)

        X = df[featuresCol].toPandas()
        y = df.select(labelCol).toPandas()
        xgboost = xgb.XGBRegressor(**best)
        xgboost = xgboost.fit(X, y)
        return XGBoostModel(labelCol=labelCol, inputCols=featuresCol, predictionCol=predCol, model=xgboost)
