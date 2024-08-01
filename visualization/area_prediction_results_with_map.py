import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Point

# Define the file paths
country_borders_path = "./visualization/cn-country-border.json"
grid_static_features_path = './dataset/grid_static_features.csv'
daily_dataset_path = './dataset/dataset_daily.csv'

# Read the country borders and grid static features data
country_borders = gpd.read_file(country_borders_path)
grid_static_features = pd.read_csv(grid_static_features_path)

# Define specific provinces to analyze
# specific_provinces = ['北京市', '天津市', '河北省', '山东省', '河南省', '安徽省', '江苏省']
specific_provinces = ['上海市', '江苏省', '浙江省', '安徽省']
selected_provinces = country_borders[country_borders['name'].isin(specific_provinces)]

# Filter grid data for all grid points in the specific provinces
specific_provinces_grid_static_features = grid_static_features[
    grid_static_features['province'].isin(specific_provinces)]


# Define grid for specific provinces
lon_min, lon_max, lat_min, lat_max = specific_provinces_grid_static_features['lon'].min(), \
specific_provinces_grid_static_features['lon'].max(), specific_provinces_grid_static_features['lat'].min(), \
specific_provinces_grid_static_features['lat'].max()
lon_specific = np.arange(lon_min, lon_max + 0.2, 0.05)
lat_specific = np.arange(lat_min, lat_max + 0.2, 0.05)

# Create new latitude and longitude grid
lon_grid_specific, lat_grid_specific = np.meshgrid(lon_specific, lat_specific)

# Linear interpolation for ELEVATION
elevation_specific_linear = griddata(
    (specific_provinces_grid_static_features['lon'], specific_provinces_grid_static_features['lat']),
    specific_provinces_grid_static_features['ELEVATION'], (lon_grid_specific, lat_grid_specific), method='linear')

# Convert to DataFrame and then GeoDataFrame for clipping
df_interpolated = pd.DataFrame({'lon': lon_grid_specific.ravel(), 'lat': lat_grid_specific.ravel(),
                                'elevation': elevation_specific_linear.ravel()})
geometry = [Point(xy) for xy in zip(df_interpolated['lon'], df_interpolated['lat'])]
gdf_interpolated = gpd.GeoDataFrame(df_interpolated, geometry=geometry, crs="EPSG:4326")

# Clip data with specific provinces borders
gdf_interpolated_clipped = gpd.overlay(gdf_interpolated, selected_provinces, how='intersection')

# Filter grid data for selected provinces
daily_dataset = pd.read_csv(daily_dataset_path)
unique_grid_ids = daily_dataset['grid_id'].unique()
filtered_grid_features = grid_static_features[
    grid_static_features['grid_id'].isin(unique_grid_ids) & grid_static_features['province'].isin(specific_provinces)]

# Divide the grid IDs into training and testing sets
train_grid_ids, test_grid_ids = train_test_split(filtered_grid_features['grid_id'].unique(), test_size=1 / 10)

pd.DataFrame(train_grid_ids, columns=['grid_id']).to_csv('YRD_train_grid_ids.csv', index=False)
pd.DataFrame(test_grid_ids, columns=['grid_id']).to_csv('YRD_test_grid_ids.csv', index=False)


# Create figure and axes for plotting
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°N'))
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°E'))

# Define colormap
cmap = plt.get_cmap('terrain')
norm = mcolors.Normalize(vmin=grid_static_features['ELEVATION'].min(),
                         vmax=grid_static_features['ELEVATION'].max())

# Plot interpolated elevation values for specific provinces
# plt.scatter(gdf_interpolated_clipped['lon'], gdf_interpolated_clipped['lat'], c=gdf_interpolated_clipped['elevation'],
#             s=2, cmap=cmap, norm=norm, alpha=.7)

# Plot test and training grid points
ax.scatter(filtered_grid_features.loc[filtered_grid_features['grid_id'].isin(test_grid_ids), 'lon'],
           filtered_grid_features.loc[filtered_grid_features['grid_id'].isin(test_grid_ids), 'lat'],
           s=4, c="red", label='Test Grids')
ax.scatter(filtered_grid_features.loc[filtered_grid_features['grid_id'].isin(train_grid_ids), 'lon'],
           filtered_grid_features.loc[filtered_grid_features['grid_id'].isin(train_grid_ids), 'lat'],
           s=4, c="#000000", label='Training Grids')


# Adjust axis tick positions
ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, prune='both'))
ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='both'))

# Plot the borders of the selected provinces
selected_provinces.plot(fc="none", ec="gray", ax=ax, lw=0.5, zorder=1.2)

# Add custom legend at the upper right corner
legend = ax.legend(loc='upper right', frameon=False, fontsize=8, handlelength=0.5)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#
# pos = ax.get_position()
# cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
# fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

# Save figure
# plt.savefig('YRD_test_and_training_map.png', format='png', transparent=True, pad_inches=0, dpi=130, bbox_inches='tight')
plt.show()
plt.close()
