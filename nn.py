import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from evaluation_metrics import symmetric_mean_absolute_percentage_error

# Constants
DATA_PATH = './dataset/dataset_daily.csv'
TRAIN_GRID_IDS_PATH = './dataset/train_grid_ids.csv'
TEST_GRID_IDS_PATH = './dataset/test_grid_ids.csv'
FEATURE_COLUMNS = ['lat', 'lon', 'TMP_P0_L1_GLL0', 'SPFH_P0_2L108_GLL0', 'RH_P0_L4_GLL0', 'PWAT_P0_L200_GLL0',
                   'UGRD_P0_L6_GLL0', 'GUST_P0_L1_GLL0', 'PRES_P0_L7_GLL0', 'CultivatedLand', 'WoodLand',
                   'GrassLand', 'Waters', 'UrbanRural', 'UnusedLand', 'Ocean', 'ELEVATION', 'AOD', 'month', 'weekday']
TARGET_COLUMN = 'o3'


# Define NN model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x



def load_and_split_data():
    # Load the full dataset
    dataset_daily = pd.read_csv(DATA_PATH)

    # Load the grid ids
    train_grid_ids = pd.read_csv(TRAIN_GRID_IDS_PATH)['grid_id']
    test_grid_ids = pd.read_csv(TEST_GRID_IDS_PATH)['grid_id']

    # Split the dataset
    train_dataset = dataset_daily[dataset_daily['grid_id'].isin(train_grid_ids)]
    test_dataset = dataset_daily[dataset_daily['grid_id'].isin(test_grid_ids)]

    return train_dataset, test_dataset


def predict_with_nn(train_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Print the device that will be used
    if device.type == 'cuda':
        print('Using GPU for training')
    else:
        print('Using CPU for training')

    # Prepare training and test data
    X_train = train_dataset[FEATURE_COLUMNS]
    y_train = train_dataset[TARGET_COLUMN]
    X_test = test_dataset[FEATURE_COLUMNS]
    y_test = test_dataset[TARGET_COLUMN]

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to tensors
    train_data = TensorDataset(torch.from_numpy(X_train.astype(np.float32)),
                               torch.from_numpy(y_train.values.astype(np.float32)))
    train_loader = DataLoader(train_data, batch_size=4092*64, shuffle=True)

    test_data = torch.from_numpy(X_test.astype(np.float32)).to(device)

    # Define model, loss and optimizer
    model = NeuralNetwork(X_train.shape[1]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the model
    for epoch in range(200):  # You can adjust the number of epochs
        loss_list = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), targets)

            # Add loss to the list
            loss_list.append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Episode {epoch}: Mean Loss = {sum(loss_list)/len(loss_list):.4f}')

    # Predict
    model.eval()
    with torch.no_grad():
        preds = model(test_data).cpu().numpy()

    # Convert 2D preds array into 1D
    preds = preds.reshape(-1)

    return y_test, preds




def evaluate_predictions(y_test, preds):
    print(f"R2 score: {r2_score(y_test, preds)}")
    print(f"MAE: {mean_absolute_error(y_test, preds)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds))}")
    print(f"Symmetric MAPE: {symmetric_mean_absolute_percentage_error(y_test, preds)}")


# Load and split data
train_dataset, test_dataset = load_and_split_data()

# Predict and evaluate
y_test, preds = predict_with_nn(train_dataset, test_dataset)
evaluate_predictions(y_test, preds)
