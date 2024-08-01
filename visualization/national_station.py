import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Point

# File paths
country_borders_path = "./visualization/cn-country-border.json"
national_station_path = './dataset/national_station.csv'

# Read country borders
country_borders = gpd.read_file(country_borders_path)

# Read grid data
df = pd.read_csv('./dataset/grid_static_features.csv')

# Create 5km grid
lon_min, lon_max, lat_min, lat_max = df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max()
lon_5km = np.arange(lon_min, lon_max + 0.1, 0.05)
lat_5km = np.arange(lat_min, lat_max + 0.1, 0.05)

# Create new latitude and longitude grid
lon_grid_5km, lat_grid_5km = np.meshgrid(lon_5km, lat_5km)

# Linear interpolation for ELEVATION
elevation_5km_linear = griddata((df['lon'], df['lat']), df['ELEVATION'], (lon_grid_5km, lat_grid_5km), method='linear')

# Convert to DataFrame
df_interpolated = pd.DataFrame({'lon': lon_grid_5km.ravel(), 'lat': lat_grid_5km.ravel(), 'elevation': elevation_5km_linear.ravel()})

# Create GeoDataFrame and set CRS
geometry = [Point(xy) for xy in zip(df_interpolated['lon'], df_interpolated['lat'])]
gdf_interpolated = gpd.GeoDataFrame(df_interpolated, geometry=geometry, crs="EPSG:4326")

# Clip data with China borders
gdf_interpolated_clipped = gpd.overlay(gdf_interpolated, country_borders, how='intersection')

# Load national station data
national_station_df = pd.read_csv(national_station_path)

# Values for color normalization
min_value = df_interpolated['elevation'].min()
max_value = df_interpolated['elevation'].max()
norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

# Setup the figure and axis with a warm gold color theme
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('#fff5e1')  # Gold tinted background for the map
fig.patch.set_facecolor('#fff5e1')

# Load China map
china_main = gpd.read_file(country_borders_path)

# Plot China borders
china_main.plot(fc="none", ec="#b08d57", ax=ax, lw=1, zorder=2)  # Border color is a shade of gold
ax.set_ylim(15, 55)
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°N'))
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°E'))

# Use the Oranges colormap for the elevation data
cmap = plt.get_cmap('Oranges')

# Plot elevation and stations
ax.scatter(gdf_interpolated_clipped['lon'], gdf_interpolated_clipped['lat'], c=gdf_interpolated_clipped['elevation'], s=2, ec="#b08d57", lw=0.1, cmap=cmap, norm=norm)
ax.scatter(national_station_df['lon'], national_station_df["lat"], s=4, color='#d93900', edgecolors='#b08d57', linewidth=0.5)  # Station dots in orange-red with gold borders

# Grid setup with subtle gold grid lines
ax.grid(which="both", linestyle="--", axis="both", c="#b08d57", linewidth=.8, alpha=.3)

# Setup the colorbar for elevation
pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax, orientation='vertical')
cbar.ax.yaxis.set_tick_params(color='#b08d57')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#b08d57')

# Adjust the appearance of axis labels for visibility
ax.tick_params(axis='x', colors='#b08d57')
ax.tick_params(axis='y', colors='#b08d57')
ax.xaxis.label.set_color('#b08d57')
ax.yaxis.label.set_color('#b08d57')

plt.savefig('ELEVATION_and_stations_gold_theme.png', format='png', transparent=True, pad_inches=0.1, dpi=150, bbox_inches='tight')
# Display the map
plt.show()
plt.close()
