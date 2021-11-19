# M5 dataset Forecasting using Pyspark

Generated monthly sales forecast of each store for a given department on M5 store-department data (https://www.kaggle.com/c/m5-forecasting-accuracy/overview). ML pipelines have been created using Spark MLLIB library. The pipeline have following modules: 

## 1) Preprocessing:
Custom Spark MLLIB transformer for
- DataAggregation Store-department level (Transformer) 
- MarkZero/NegativeSales
- ImputeMean (store) 
- Train_Test Split

## 2) Feature Engineering
Custom Spark MLLIB transformer for
- LagFeature
- LogTransformation
- Scaling

## 3) Model Training 
Custom Spark MLLIB estimator for
- Random Forrest
- XGBoost (xgboost library)
- FbProphet

 ### Parameter optimization
- Hyper parameter tunning using Hyperopt 
- Re-training using best parameters

## 4) Evaluator
Custom Spark MLLIB evaluator for
- MAPE evaluation matrix
