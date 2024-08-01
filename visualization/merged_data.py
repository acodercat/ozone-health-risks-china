import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv('./dataset/grid_features.csv')

dates = ['2021-03-01', '2021-06-01', '2021-10-01', '2021-11-30']


data = data[data['date'].isin(dates)]

# 确保经纬度列是数值类型
data['lat'] = data['lat'].astype(float)
data['lon'] = data['lon'].astype(float)

# 定义需要聚合的数据列
columns_to_aggregate = ['TMP_P0_L1_GLL0', 'SPFH_P0_2L108_GLL0', 'RH_P0_L4_GLL0', 'PWAT_P0_L200_GLL0', 'UGRD_P0_L6_GLL0',
                        'GUST_P0_L1_GLL0', 'PRES_P0_L7_GLL0', 'CultivatedLand', 'WoodLand', 'GrassLand', 'Waters',
                        'UrbanRural', 'UnusedLand', 'Ocean', 'ELEVATION', 'AOD']

# 将经纬度按0.25间隔四舍五入到最近的网格中心点，这样可以将数据按50公里网格大小分组
# 由于原始网格大小为25km，所以合并四个网格到50km，经纬度增加0.25即可定位到新的网格中心
data['grouped_lat'] = ((data['lat'] // 0.5) * 0.5) + 0.25
data['grouped_lon'] = ((data['lon'] // 0.5) * 0.5) + 0.25

# 分组并计算每个新网格的平均值
grouped_data = data.groupby(['date', 'grouped_lat', 'grouped_lon'])
averaged_data = grouped_data[columns_to_aggregate].mean().reset_index()

# 输出到CSV文件
averaged_data.to_csv('path_to_your_new_file.csv', index=False)

print("数据已按照50公里*50公里网格处理完毕，并包含了日期信息。")