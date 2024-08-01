import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from evaluation_metrics import symmetric_mean_absolute_percentage_error
import numpy as np

# Constants
DATASET_DAILY_PATH = './dataset/dataset_daily.csv'
TRAIN_GRID_IDS_PATH = './dataset/train_grid_ids.csv'
TEST_GRID_IDS_PATH = './dataset/test_grid_ids.csv'

FEATURE_COLUMNS = ['lat', 'lon', 'TMP_P0_L1_GLL0', 'SPFH_P0_2L108_GLL0', 'RH_P0_L4_GLL0',
                   'PWAT_P0_L200_GLL0', 'UGRD_P0_L6_GLL0', 'GUST_P0_L1_GLL0',
                   'PRES_P0_L7_GLL0', 'CultivatedLand', 'WoodLand', 'GrassLand', 'Waters',
                   'UrbanRural', 'UnusedLand', 'Ocean', 'ELEVATION', 'AOD', 'month',
                   'weekday']

TARGET_COLUMN = 'o3'



def load_and_split_data():
    # Load the full dataset
    dataset_daily = pd.read_csv(DATASET_DAILY_PATH)

    # Load the grid ids
    train_grid_ids = pd.read_csv(TRAIN_GRID_IDS_PATH)['grid_id']
    test_grid_ids = pd.read_csv(TEST_GRID_IDS_PATH)['grid_id']

    # Split the dataset
    train_dataset = dataset_daily[dataset_daily['grid_id'].isin(train_grid_ids)]
    test_dataset = dataset_daily[dataset_daily['grid_id'].isin(test_grid_ids)]

    return train_dataset, test_dataset


def create_and_fit_grid_search(train_dataset, test_dataset):
    # Prepare training and test data
    X_train = train_dataset[FEATURE_COLUMNS]
    y_train = train_dataset[TARGET_COLUMN]
    X_test = test_dataset[FEATURE_COLUMNS]
    y_test = test_dataset[TARGET_COLUMN]

    # Define the pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    # Define the grid of parameters to search
    param_grid = {
        'svr__C': [0.1, 1, 10, 100, 1000],
        'svr__gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        'svr__kernel': ['rbf']
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit the GridSearchCV object
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print("Best parameters: ", grid_search.best_params_)

    # Make predictions
    daily_predictions = grid_search.predict(X_test)

    return y_test, daily_predictions


def evaluate_predictions(y_test, preds):
    print(f"R2 score: {r2_score(y_test, preds)}")
    print(f"MAE: {mean_absolute_error(y_test, preds)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds))}")
    print(f"Symmetric MAPE: {symmetric_mean_absolute_percentage_error(y_test, preds)}")


# Load and split data
train_dataset, test_dataset = load_and_split_data()

# Create GridSearch, fit and predict
y_test, preds = create_and_fit_grid_search(train_dataset, test_dataset)

# Evaluate
evaluate_predictions(y_test, preds)
