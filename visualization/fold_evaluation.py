import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Your functions remain the same

def calculate_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred))) * 200


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


actual_pred_dfs = [pd.read_csv(f"./dataset/folds/test_set_performance_{i}.csv") for i in range(10)]

global_min = float('inf')
global_max = float('-inf')
for df in actual_pred_dfs:
    df = df[df['pred'] <= 250]
    df = df[df['actual'] <= 250]
    local_min = min(df['actual'].min(), df['pred'].min())
    local_max = max(df['actual'].max(), df['pred'].max())
    global_min = min(global_min, local_min)
    global_max = max(global_max, local_max)

for idx, df in enumerate(actual_pred_dfs):
    df = df[df['pred'] <= 250]
    df = df[df['actual'] <= 250]
    actual, pred = df["actual"], df["pred"]

    grid = sns.jointplot(x=actual, y=pred, kind='hex', height=3)  # Adjust height as needed

    # 使背景透明
    grid.ax_joint.set_facecolor((0, 0, 0, 0))
    grid.ax_marg_x.set_facecolor((0, 0, 0, 0))
    grid.ax_marg_y.set_facecolor((0, 0, 0, 0))
    grid.fig.set_facecolor((0, 0, 0, 0))

    # 使 hexbin 背景透明
    cmap = plt.cm.Blues
    cmap._init()
    alphas = np.linspace(0, 1, cmap.N + 3)
    cmap._lut[:, -1] = alphas

    grid.ax_joint.collections[0].set_cmap(cmap)

    grid.ax_joint.plot([global_min, global_max], [global_min, global_max], 'k--', lw=1)
    grid.ax_joint.set_xlim(global_min, global_max)
    grid.ax_joint.set_ylim(global_min, global_max)

    actual_np = np.array(actual)
    slope, _, _, _ = np.linalg.lstsq(actual_np[:, np.newaxis], pred, rcond=None)
    slope = slope[0]

    grid.ax_joint.plot(actual, slope * actual, 'r', lw=1)

    rmse = calculate_rmse(actual, pred)
    r2_daily = r2_score(actual, pred)
    grid.ax_joint.text(0.02, 0.98, f'$RMSE: {rmse:.2f}$\n$R^2: {r2_daily:.2f}$', transform=grid.ax_joint.transAxes,
                       fontsize=8, va='top')
    grid.ax_joint.text(0.90, 0.98, f'Fold {idx + 1}', transform=grid.ax_joint.transAxes, ha='right', va='top',
                       fontweight='bold')

    # Save each jointplot to an individual file
    grid.savefig(f"fold_{idx + 1}_evaluation.png", format='png', transparent=True)

    # If you want to show each plot one by one
    # plt.show()
