import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


TRAIN_GRID_IDS_PATH = './dataset/train_grid_ids.csv'
TEST_GRID_IDS_PATH = './dataset/test_grid_ids.csv'
FEATURE_COLUMNS = ['lat', 'lon', 'TMP_P0_L1_GLL0', 'SPFH_P0_2L108_GLL0', 'RH_P0_L4_GLL0', 'PWAT_P0_L200_GLL0',
                   'UGRD_P0_L6_GLL0', 'GUST_P0_L1_GLL0', 'PRES_P0_L7_GLL0', 'CultivatedLand', 'WoodLand',
                   'GrassLand', 'Waters', 'UrbanRural', 'UnusedLand', 'Ocean', 'ELEVATION', 'AOD', 'month', 'weekday']
TARGET_COLUMN = 'o3'


def predict_daily(train_dataset, test_dataset):

    X_train = train_dataset[FEATURE_COLUMNS]
    y_train = train_dataset[TARGET_COLUMN]
    X_test = test_dataset[FEATURE_COLUMNS]
    y_test = test_dataset[TARGET_COLUMN]



    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # print('features:', len(X_train))

    # xgb_params = {
    #     'booster': 'gbtree',
    #     'objective': 'reg:squarederror',
    #     'eval_metric': 'rmse',
    #     'eta': 0.05,
    #     'max_depth': 35,
    #     'learning_rate': 0.02,
    # }

    # evallist = [(dtest, 'eval'), (dtrain, 'train')]

    # bst = xgb.train(xgb_params, dtrain, num_boost_round=100, evals=evallist)

    # preds = bst.predict(dtest)
    xgb_model = xgb.XGBRegressor(learning_rate=0.01,max_depth=25,n_estimators=500)
    xgb_model.fit(X_train, y_train)
    preds = xgb_model.predict(X_test)
    return y_test, preds




DATA_PATH = './dataset/dataset_daily.csv'
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



# Load and split data
train_dataset, test_dataset = load_and_split_data()

# Predictions for daily and monthly datasets
daily_y_test, daily_predictions = predict_daily(train_dataset, test_dataset)

# rmse_daily = symmetric_mean_absolute_percentage_error(daily_y_test, daily_predictions)
r2_daily = r2_score(daily_y_test, daily_predictions)


# 提取actual和pred列
actual = daily_y_test
pred = daily_predictions

# 计算RMSE
rmse = root_mean_squared_error(actual, pred)
print(f"RMSE: {rmse:.4f}")

# 计算MAE 
mae = mean_absolute_error(actual, pred)
print(f"MAE: {mae:.4f}")

# 计算NMB
nmb = (pred.sum() - actual.sum()) / actual.sum()
print(f"NMB: {nmb:.4f}")

# 计算NME
nme = (abs(pred - actual)).sum() / actual.sum() 
print(f"NME: {nme:.4f}")


r2 = r2_score(actual, pred)

print(f"R2: {r2:.4f}")



# monthly_y_test, monthly_predictions = predict_monthly(monthly_dataset, target_column, monthly_feature_columns)

# # Create two subplots without shared y-axis
# fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# # Scatterplot for daily predictions
# sns.scatterplot(x=daily_y_test, y=daily_predictions, color="skyblue", edgecolor="darkblue", alpha=0.6, ax=axs[0])
# # Title for the daily scatterplot
# axs[0].set_title('Daily Prediction', fontsize=16)
# # Diagonal line
# axs[0].plot([daily_y_test.min(), daily_y_test.max()], [daily_y_test.min(), daily_y_test.max()], 'k--', lw=3)


# # 使用 lstsq 进行线性拟合，不包含截距
# slope, _, _, _ = np.linalg.lstsq(daily_y_test[:, np.newaxis], daily_predictions, rcond=None)
# slope = slope[0]


# axs[0].plot(daily_y_test, slope * daily_y_test, 'r', lw=2)
# # x and y labels for daily scatterplot
# axs[0].set_xlabel('Actual PM$_{2.5}$ (ug/m$^3$)', fontsize=12)
# axs[0].set_ylabel('Predicted PM$_{2.5}$ (ug/m$^3$)', fontsize=12)
# axs[0].text(0.98, 0.02, '(a)', transform=axs[0].transAxes, fontsize=16, va='bottom', ha='right')
# # Add RMSE and R^2 to the plot

# axs[0].text(0.02, 0.98, f'$RMSE: {rmse_daily:.2f}$\n$R^2: {r2_daily:.2f}$', transform=axs[0].transAxes, fontsize=12,
#             va='top')

# # Insert a line subplot for daily predictions at the top-left corner of the scatterplot
# # daily_index = np.arange(len(daily_y_test))
# # inset_axes_daily = inset_axes(axs[0], width="40%", height="20%", loc=2, bbox_to_anchor=(.2, 0, 0, 0), bbox_transform=axs[0].transAxes)
# # inset_axes_daily.plot(daily_index, daily_y_test, color='skyblue', label='Actual')
# # inset_axes_daily.plot(daily_index, daily_predictions, color='darkblue', label='Predicted')
# # inset_axes_daily.legend(fontsize=6)
# # # inset_axes_daily.set_xlabel('Times', fontsize=6)
# # # inset_axes_daily.set_ylabel('Actual values', fontsize=6)
# # inset_axes_daily.tick_params(axis='both', which='major', labelsize=6)

# # Scatterplot for monthly predictions
# sns.scatterplot(x=monthly_y_test, y=monthly_predictions, color="red", edgecolor="darkblue", alpha=0.6, ax=axs[1])
# # Diagonal line
# axs[1].plot([monthly_y_test.min(), monthly_y_test.max()], [monthly_y_test.min(), monthly_y_test.max()], 'k--', lw=3)


# slope, _, _, _ = np.linalg.lstsq(monthly_y_test[:, np.newaxis], monthly_predictions, rcond=None)
# slope = slope[0]


# axs[1].plot(monthly_y_test, slope * monthly_y_test, 'r', lw=2)

# # Title for the monthly scatterplot
# axs[1].set_title('Monthly Prediction', fontsize=16)
# # x and y labels for monthly scatterplot
# axs[1].set_xlabel('Actual PM$_{2.5}$ (ug/m$^3$)', fontsize=12)
# axs[1].set_ylabel('Predicted PM$_{2.5}$ (ug/m$^3$)', fontsize=12)
# axs[1].text(0.98, 0.02, '(b)', transform=axs[1].transAxes, fontsize=16, va='bottom', ha='right')
# rmse_monthly = symmetric_mean_absolute_percentage_error(monthly_y_test, monthly_predictions)
# r2_monthly = r2_score(monthly_y_test, monthly_predictions)
# axs[1].text(0.02, 0.98, f'$RMSE: {rmse_monthly:.2f}$\n$R^2: {r2_monthly:.2f}$', transform=axs[1].transAxes,
#             fontsize=12, va='top')

# # Insert a line subplot for monthly predictions at the top-left corner of the scatterplot
# # monthly_index = np.arange(len(monthly_y_test))
# # inset_axes_monthly = inset_axes(axs[1], width="40%", height="20%", loc=2)
# # inset_axes_monthly.plot(monthly_index, monthly_y_test, color='skyblue', label='Actual')
# # inset_axes_monthly.plot(monthly_index, monthly_predictions, color='darkblue', label='Predicted')
# # inset_axes_monthly.legend(fontsize=6)
# # # inset_axes_monthly.set_xlabel('Times', fontsize=6)
# # # inset_axes_monthly.set_ylabel('Actual values', fontsize=6)
# # inset_axes_monthly.tick_params(axis='both', which='major', labelsize=6)

# # Adjust spaces between subplots
# # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
# plt.subplots_adjust(left=None, right=None)

# # plt.savefig('xgb.png', format='png', transparent=True, dpi=130, pad_inches=0.1, bbox_inches='tight')

# plt.show()
