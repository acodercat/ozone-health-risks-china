import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
from evaluation_metrics import symmetric_mean_absolute_percentage_error

# Constants
DATA_PATH = './dataset/dataset_daily.csv'
TRAIN_GRID_IDS_PATH = './dataset/train_grid_ids.csv'
TEST_GRID_IDS_PATH = './dataset/test_grid_ids.csv'
FEATURE_COLUMNS = ['lat', 'lon', 'TMP_P0_L1_GLL0', 'SPFH_P0_2L108_GLL0', 'RH_P0_L4_GLL0',
                   'PWAT_P0_L200_GLL0', 'UGRD_P0_L6_GLL0', 'GUST_P0_L1_GLL0',
                   'PRES_P0_L7_GLL0', 'CultivatedLand', 'WoodLand', 'GrassLand', 'Waters',
                   'UrbanRural', 'UnusedLand', 'Ocean', 'ELEVATION', 'AOD', 'month', 'weekday']
TARGET_COLUMN = 'o3'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    # Load the full dataset
    dataset_daily = pd.read_csv(DATA_PATH)

    # Load the grid ids
    train_grid_ids = pd.read_csv(TRAIN_GRID_IDS_PATH)['grid_id']
    test_grid_ids = pd.read_csv(TEST_GRID_IDS_PATH)['grid_id']

    # Split the dataset
    train_dataset = dataset_daily[dataset_daily['grid_id'].isin(train_grid_ids)]
    test_dataset = dataset_daily[dataset_daily['grid_id'].isin(test_grid_ids)]

    return train_dataset, test_dataset


def preprocess_data(train_dataset, test_dataset, feature_columns, target_column):
    # Prepare training data
    X_train = train_dataset[feature_columns].values
    y_train = train_dataset[target_column].values

    # Prepare test data
    X_test = test_dataset[feature_columns].values
    y_test = test_dataset[target_column].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


def train_nn_model(X_train_scaled, y_train):
    # Define NN model
    class NeuralNetwork(nn.Module):
        def __init__(self, input_size):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    nn_model = NeuralNetwork(X_train_scaled.shape[1]).to(device)

    # Convert data to tensors
    train_data = TensorDataset(torch.from_numpy(X_train_scaled.astype(np.float32)),
                               torch.from_numpy(y_train.astype(np.float32)))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nn_model.parameters())

    # Train the model
    for epoch in range(10):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            outputs = nn_model(inputs)
            loss = criterion(outputs.view(-1), targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return nn_model


def train_meta_model(X_train, y_train):
    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 32768,
        'max_depth': 15,
        'max_bin': 100,
        'min_data_in_leaf': 500,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    # Meta learner
    lgb_train = lgb.Dataset(X_train, y_train)
    meta_model = lgb.train(lgbm_params, lgb_train)

    return meta_model


def evaluate_model(y_test, predictions):
    print(f"R2 score: {r2_score(y_test, predictions)}")
    print(f"MAE: {mean_absolute_error(y_test, predictions)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}")
    print(f"Symmetric MAPE: {symmetric_mean_absolute_percentage_error(y_test, predictions)}")


# Load data
train_dataset, test_dataset = load_data()

# Preprocess data
X_train_scaled, y_train, X_test_scaled, y_test = preprocess_data(train_dataset, test_dataset, FEATURE_COLUMNS,
                                                                 TARGET_COLUMN)

# Train base learner (NN model)
nn_model = train_nn_model(X_train_scaled, y_train)

# Collect predictions from base learner
base_preds_train = nn_model(torch.from_numpy(X_train_scaled.astype(np.float32)).to(device)).cpu().numpy().ravel()
base_preds_test = nn_model(torch.from_numpy(X_test_scaled.astype(np.float32)).to(device)).cpu().numpy().ravel()

# Train meta learner (LightGBM model)
meta_model = train_meta_model(base_preds_train.reshape(-1, 1), y_train)

# Predict
final_predictions = meta_model.predict(base_preds_test.reshape(-1, 1))

# Evaluate
evaluate_model(y_test, final_predictions)
