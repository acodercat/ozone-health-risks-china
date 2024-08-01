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

# Read country borders
country_borders = gpd.read_file(country_borders_path)

# Read grid data
df = pd.read_csv('./dataset/grid_static_features.csv')

# Create 5km grid
lon_min, lon_max, lat_min, lat_max = df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max()
lon_5km = np.arange(lon_min, lon_max + 0.1, 0.05)
lat_5km = np.arange(lat_min, lat_max + 0.1, 0.05)

definition = [{
    "field_name": 'GrassLand',
    "name": '草地覆盖',
    "unit": 'km\u00b2',
    "cmap": 'Greens'
},
    {
        "field_name": 'CultivatedLand',
        "name": '农耕用地',
        "unit": 'km\u00b2',
        "cmap": 'YlOrBr'
    },
    {
        "field_name": 'WoodLand',
        "name": '林地覆盖',
        "unit": 'km\u00b2',
        "cmap": 'viridis'
    },
    {
        "field_name": 'Waters',
        "name": '水体覆盖',
        "unit": 'km\u00b2',
        "cmap": 'Blues'
    },
    {
        "field_name": 'UrbanRural',
        "name": '城市-乡村比例',
        "unit": '',
        "cmap": 'coolwarm'
    },
    {
        "field_name": 'UnusedLand',
        "name": '未用地',
        "unit": 'km\u00b2',
        "cmap": 'gray'
    },
    {
        "field_name": 'Ocean',
        "name": '海洋覆盖',
        "unit": 'km\u00b2',
        "cmap": 'ocean'
    },
    {
        "field_name": 'ELEVATION',
        "name": '海拔',
        "unit": 'km\u00b2',
        "cmap": 'terrain'
    }
]

current_index = 7

# Create new latitude and longitude grid
lon_grid_5km, lat_grid_5km = np.meshgrid(lon_5km, lat_5km)

# Linear interpolation for ELEVATION
elevation_5km_linear = griddata((df['lon'], df['lat']), df[definition[current_index]['field_name']],
                                (lon_grid_5km, lat_grid_5km), method='linear')

# Convert to DataFrame
df_interpolated = pd.DataFrame(
    {'lon': lon_grid_5km.ravel(), 'lat': lat_grid_5km.ravel(), 'value': elevation_5km_linear.ravel()})

# Create GeoDataFrame and set CRS
geometry = [Point(xy) for xy in zip(df_interpolated['lon'], df_interpolated['lat'])]
gdf_interpolated = gpd.GeoDataFrame(df_interpolated, geometry=geometry, crs="EPSG:4326")

# Clip data with China borders
gdf_interpolated_clipped = gpd.overlay(gdf_interpolated, country_borders, how='intersection')

# Load China map
china_main = gpd.read_file(country_borders_path)

# Values for color normalization

min_value = df_interpolated['value'].min()
max_value = df_interpolated['value'].max()

fig, ax = plt.subplots(gridspec_kw={'right': 0.87}, dpi=130)
# fig, ax = plt.subplots(figsize=(12, 8))
# plt.tight_layout()

# Plot China borders
china_main.plot(fc="none", ec="grey", ax=ax, lw=0.5, zorder=2)
ax.set_ylim(15, 55)
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°N'))
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°E'))

# Colormap setup
cmap = plt.get_cmap(definition[current_index]['cmap'])
norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

# Plot elevation and stations
ax.scatter(gdf_interpolated_clipped['lon'], gdf_interpolated_clipped['lat'], c=gdf_interpolated_clipped['value'],
           s=2, ec="b", lw=0, cmap=cmap, norm=norm, alpha=0.5)

# Grid setup
ax.grid(which="both", linestyle="--", axis="both", c="gray", linewidth=.8, alpha=.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ax.set_title(definition[current_index]['name'], fontsize=14)

# fig.canvas.draw()
# Colorbar setup
pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

# cbar.set_label(f'{definition[current_index]["name"]}（{definition[current_index]["unit"]}）', rotation=270, labelpad=12,
#                fontsize=10)
# plt.tight_layout()
# Save figure
plt.savefig(f'{definition[current_index]["field_name"]}.png', format='png', transparent=True, pad_inches=0.1, dpi=130,
            bbox_inches='tight')
plt.show()
plt.close()
