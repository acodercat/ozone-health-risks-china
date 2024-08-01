import pandas as pd

grid_static_features = pd.read_csv('./dataset/grid_static_features.csv')
province = ['新疆维吾尔自治区']
selected_provinces = grid_static_features[grid_static_features['province'].isin(province)]
selected_provinces['grid_id'].to_csv(f'新疆维吾尔自治区.csv', index=False)

print(selected_provinces)
