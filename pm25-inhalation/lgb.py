import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred))) * 2 * 100


def smape(y_true, y_pred):
    return np.abs((y_true - y_pred) / (y_true + y_pred)) * 2 * 100


def predict_daily(daily_dataset, train_grid_ids, test_grid_ids, target_column, feature_columns):
    # daily_dataset = daily_dataset[daily_dataset[target_column] <= 300]

    train_dataset = daily_dataset[daily_dataset['grid_id'].isin(train_grid_ids)]
    test_dataset = daily_dataset[daily_dataset['grid_id'].isin(test_grid_ids)]
    
    X_train = train_dataset[feature_columns]
    y_train = train_dataset[target_column]

    X_test = test_dataset[feature_columns]
    y_test = test_dataset[target_column]



    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    print('features:', len(X_train))

    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 1500,
        'max_depth': 23,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 4,
        'verbose': -1
    }

    lgbm = lgb.train(lgb_params, lgb_train, num_boost_round=1800, valid_sets=lgb_test)
    lgbm.save_model('lgbm_model.bin')
    preds = lgbm.predict(X_test, num_iteration=lgbm.best_iteration)
    return y_test, preds

def evaluate_predictions(y_test, preds):
    print(f"R2 score: {r2_score(y_test, preds)}")
    print(f"MAE: {mean_absolute_error(y_test, preds)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds))}")
    print(f"Symmetric MAPE: {symmetric_mean_absolute_percentage_error(y_test, preds)}")

daily_dataset = pd.read_csv('./dataset_daily.csv')


train_grid_ids = pd.read_csv('./train_grid_ids.csv')
test_grid_ids = pd.read_csv('./test_grid_ids.csv')


target_column = 'pm2_5'

daily_feature_columns = ['lat', 'lon',
                         'TMP_P0_L1_GLL0', 'SPFH_P0_2L108_GLL0', 'RH_P0_L4_GLL0',
                         'PWAT_P0_L200_GLL0', 'UGRD_P0_L6_GLL0', 'GUST_P0_L1_GLL0',
                         'PRES_P0_L7_GLL0', 'CultivatedLand', 'WoodLand', 'GrassLand', 'Waters',
                         'UrbanRural', 'UnusedLand', 'Ocean', 'ELEVATION', 'AOD', 'month',
                         'year', 'weekday', 'population']




daily_y_test, daily_predictions = predict_daily(daily_dataset, train_grid_ids['grid_id'], test_grid_ids['grid_id'], target_column, daily_feature_columns)


evaluate_predictions(daily_y_test, daily_predictions)

grid_features = pd.read_csv('./grid_features.csv')

X = grid_features[daily_feature_columns]

lgbm = lgb.Booster(model_file='lgbm_model.bin')

preds = lgbm.predict(X)

assert len(preds) == len(X)

grid_features['pm2_5'] = preds


result_df = grid_features[['grid_id','lat', 'lon', 'population', 'population_weight', 'date', 'pm2_5']]


# result_df.to_csv('lgb_prediction_results.csv', index=False)
