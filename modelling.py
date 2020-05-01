import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer


def train(new_data = None):
    df = pd.read_csv('train-data/train-prepared.csv')
    
    if new_data != None:
        df = df.append(new_data)
    
    df.date = pd.to_datetime(df.date)

    df = df[~df['target'].isna()]

    # train/test split
    X = df.drop(['target','country', 'date'], axis = 1)
    y = df['target']

    X_train = X # since we are now training on the whole data

    # pre-processing
    numeric_cols = list(X_train)

    preprocess = make_column_transformer(
        (StandardScaler(), numeric_cols)
    )

    X_train = preprocess.fit_transform(X_train)

    # model training
    model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=0.9, gamma=0,
                 importance_type='gain', learning_rate=0.05, max_delta_step=0,
                 max_depth=5, min_child_weight=1, missing=None, n_estimators=100,
                 n_jobs=1, nthread=None, objective='reg:squarederror',
                 random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 seed=42, silent=None, subsample=0.7, verbosity=1)

    filename = 'models/trained-model.sav'
    pickle.dump(model, open(filename, 'wb'))

    print("Model trained successfully!")

def predict(X_predict):
    # pre-processing
    numeric_cols = list(X_predict)

    preprocess = make_column_transformer(
        (StandardScaler(), numeric_cols)
    )

    X_predict = preprocess.fit_transform(X_predict)

    # predicting
    model = joblib.load(filename)
    y = model.predict(X_predict)

    return(y)


if __name__ == '__main__':
    train()

