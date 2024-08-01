import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred))) * 2 * 100

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

actual_vs_predicted = pd.read_csv("./dataset/results/folds/YRD_performane1.csv")
# actual_vs_predicted = pd.read_csv("./dataset/YRD_prediction_results.csv")

actual_values = actual_vs_predicted["actual"]
predicted_values = actual_vs_predicted["pred"]

# Create two subplots without shared y-axis
# fig, ax = plt.subplots(figsize=(8, 6))

# Scatterplot for daily predictions
joint_grid = sns.jointplot(x=actual_values, y=predicted_values, kind="hex")
# 使背景透明
joint_grid.ax_joint.set_facecolor((0, 0, 0, 0))
joint_grid.ax_marg_x.set_facecolor((0, 0, 0, 0))
joint_grid.ax_marg_y.set_facecolor((0, 0, 0, 0))
joint_grid.fig.set_facecolor((0, 0, 0, 0))

# 使 hexbin 背景透明
cmap = plt.cm.Blues
cmap._init()
alphas = np.linspace(0, 1, cmap.N + 3)
cmap._lut[:, -1] = alphas

joint_grid.ax_joint.collections[0].set_cmap(cmap)

# 获取轴的当前限制
x_lim = plt.xlim()
y_lim = plt.ylim()

# 计算text的位置
text_x = x_lim[0] + 0.08 * (x_lim[1] - x_lim[0])
text_y = y_lim[1] - 0.04 * (y_lim[1] - y_lim[0])

plt.plot([actual_values.min(), actual_values.max()], [actual_values.min(), actual_values.max()], 'k--', lw=3)
rmse = calculate_rmse(actual_values, predicted_values)
r2_daily = r2_score(actual_values, predicted_values)


# R^2 line
actual_np = np.array(actual_values)

# 使用 lstsq 进行线性拟合，不包含截距
slope, _, _, _ = np.linalg.lstsq(actual_np[:, np.newaxis], predicted_values, rcond=None)
slope = slope[0]

plt.plot(actual_values, slope * actual_values, 'r', lw=2)

# x and y labels for daily scatterplot
joint_grid.ax_joint.set_xlabel('Actual O$_{3}$ (ug/m$^3$)', fontsize=12)
joint_grid.ax_joint.set_ylabel('Predicted O$_{3}$ (ug/m$^3$)', fontsize=12)

# 使用计算的位置放置text
plt.text(text_x, text_y, f'$RMSE: {rmse:.2f}$\n$R^2: {r2_daily:.2f}$', fontsize=12, va='top')


plt.savefig('./scatter_images/YRD_fold_evaluation_scatter.png', format='png', transparent=True, pad_inches=0.1, dpi=150)
plt.show()