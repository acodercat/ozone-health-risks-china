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
df = pd.read_csv('./grid_static_features.csv')

# Create 5km grid
lon_min, lon_max, lat_min, lat_max = df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max()
lon_5km = np.arange(lon_min, lon_max + 0.1, 0.05)
lat_5km = np.arange(lat_min, lat_max + 0.1, 0.05)

definition = [{
    "field_name": 'population',
    "name": 'population',
    "unit": 'number',
    "cmap": 'terrain'
}
]

current_index = 0



# Load China map
china_main = gpd.read_file(country_borders_path)

# Values for color normalization

min_value = df['population'].min()
max_value = df['population'].max()

# 分段的边界
breaks = [
    min_value,
    np.percentile(df['population'], 25),
    np.percentile(df['population'], 50),
    np.percentile(df['population'], 75),
    max_value
]
colors = ["#f2f0f7", "#dadaeb", "#bcbddc", "#9e9ac8", "#756bb1", "#54278f"]






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
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(breaks, cmap.N)

# Plot elevation and stations
ax.scatter(df['lon'], df['lat'], c=df['population'],
           s=2, ec="b", lw=0, cmap=cmap, norm=norm, alpha=0.5)

# Grid setup
ax.grid(which="both", linestyle="--", axis="both", c="gray", linewidth=.8, alpha=.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# ax.set_title(definition[current_index]['name'], fontsize=14)

# fig.canvas.draw()
# Colorbar setup
# pos = ax.get_position()
# cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])

# cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

# cbar.set_label(f'{definition[current_index]["name"]}（{definition[current_index]["unit"]}）', rotation=270, labelpad=12,
#                fontsize=10)
# plt.tight_layout()
# Save figure
plt.savefig(f'{definition[current_index]["field_name"]}.png', format='png', transparent=True, pad_inches=0.1, dpi=130,
            bbox_inches='tight')
plt.show()
plt.close()
