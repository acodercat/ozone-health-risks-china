import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
# 读取CSV数据
data = pd.read_csv('./dataset/dataset_daily.csv')  # 请将'your_data.csv'替换为您的实际CSV文件名

# 选择需要的列
X = data['TMP_P0_L1_GLL0'].values.reshape(-1, 1)  # 温度作为自变量X
y = data['o3'].values.reshape(-1, 1)  # 臭氧浓度作为因变量y

# 创建线性回归模型
reg = LinearRegression()
reg.fit(X, y)

# 预测值
y_pred = reg.predict(X)

# # 绘制散点图和回归线
# plt.figure(figsize=(10, 6))
# # plt.scatter(X, y, color='blue', label='Data Points')
import numpy as np

# sns.scatterplot(x=X.ravel(), y=y.ravel(), color="skyblue", edgecolor="darkblue", alpha=0.6)
# plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line',)


# plt.xlabel('Temperature (°C)')
# plt.ylabel('Ozone Concentration (ug/m$^3$)')
# # plt.title('Linear Regression: Temperature vs Ozone Concentration')
# plt.legend()

# # 显示图形
# plt.tight_layout()
# plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.figure(figsize=(8, 6))

# 创建自定义的颜色映射
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#DAEBFC", "#1877B3"])

sns.scatterplot(x=X.ravel(), y=y.ravel(), 
                c=y.ravel(),
                cmap=cmap,
                edgecolor="darkblue", 
                alpha=0.2,
                linewidth=0.1,
                s=10)

plt.plot(X, y_pred, color='#ff7f0e', linewidth=2, linestyle='--', label='Regression Line')

plt.text(0.01, 0.85, f'Coefficient (Slope): = {reg.coef_[0][0]:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
plt.text(0.01, 0.80, f'Intercept = {reg.intercept_[0]:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

plt.xlabel('Temperature (°C)', fontsize=14)
plt.ylabel('Ozone Concentration (μg/m$^3$)', fontsize=14)
# plt.title('Temperature vs Ozone Concentration', fontsize=16)

plt.legend(fontsize=12, loc='upper left')
plt.tick_params(axis='both', which='major', labelsize=12)

plt.grid(True, linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig('temp_vs_o3_improved.png', format='png', transparent=True, dpi=300, pad_inches=0.1, bbox_inches='tight')

plt.show()


# 打印回归系数和截距
print('Coefficient (Slope):', reg.coef_[0][0])
print('Intercept:', reg.intercept_[0])