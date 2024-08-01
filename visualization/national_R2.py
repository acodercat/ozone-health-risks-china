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
# national_station_path = './dataset/national_station.csv'

# Read country borders
country_borders = gpd.read_file(country_borders_path)

# Read grid data
national_R2 = pd.read_csv('./dataset/national_R2.csv')
#
# gird_static_features = pd.read_csv('./dataset/grid_static_features.csv')
#
# gird_static_features = gird_static_features[['grid_id', 'province']]
#
#
# merged_national_R2 = pd.merge(national_R2, gird_static_features, on='grid_id', how='left')
#
# print(merged_national_R2)






# Load China map
china_main = gpd.read_file(country_borders_path)

# Values for color normalization
min_value = national_R2['r2'].min()
max_value = national_R2['r2'].max()

fig, ax = plt.subplots()

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
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

# Plot elevation and stations
ax.scatter(national_R2['lon'], national_R2['lat'], c=national_R2['r2'], s=5, ec="b", lw=0, cmap=cmap, norm=norm)


# Grid setup
ax.grid(which="both", linestyle="--", axis="both", c="gray", linewidth=.8, alpha=.3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

# Colorbar setup
pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

# Save figure
plt.savefig('national_R2.png', format='png', transparent=True, pad_inches=0, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
