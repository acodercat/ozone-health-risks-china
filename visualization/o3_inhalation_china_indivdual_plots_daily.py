import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import geopandas as gpd
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np




prediction = pd.read_csv("./dataset/prediction_results.csv")
prediction['date'] = pd.to_datetime(prediction['date'].astype(str), format='%Y%m%d')
country_borders = gpd.read_file("./visualization/cn-country-border.json")
grid_static_features = pd.read_csv("./population-process/grid_static_features_with_population.csv")

grid_static_features = grid_static_features[['grid_id', 'population']]

dates = [
    '2021-08-1', '2021-08-17', '2021-08-18', 
    '2021-08-22', '2021-08-23', '2021-08-24', 
    '2021-08-25', '2021-08-26', '2021-08-27', 
]

prediction = prediction[prediction['date'].isin(dates)]

results = pd.merge(prediction, grid_static_features, on='grid_id')

results['inhalation'] = results['pred_o3']

# results.to_csv('./inhalation_o3.csv', index=False)
colors = [
    '#f7fbff',  # 浅蓝色
    '#deebf7',  # 蓝色
    '#c6dbef',  # 淡蓝色
    '#9ecae1',  # 淡青色
    '#6baed6',  # 亮蓝色
    '#4292c6',  # 中蓝色
    '#2171b5',  # 标准蓝色
    '#08519c',  # 深蓝色
    '#08306b'   # 极深蓝色
]

cmap_name = 'my_custom_cmap'
cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=2000)
# cmap = plt.get_cmap('RdYlBu_r')
vmax = results['inhalation'].quantile(0.98)
cmap = plt.get_cmap('RdYlBu_r')
norm = mcolors.Normalize(vmin=results['inhalation'].min(), vmax=vmax)
fig, axs = plt.subplots(2, 3, figsize=(8, 4), dpi=130)
axs = axs.ravel()  # Flatten array of axes
for ax, date in zip(axs, dates):
    one_day_grid_features = results[results['date'] == date]
    country_borders.plot(fc="none", ec="gray", ax=ax, lw=0.2, zorder=1.2)
    ax.tick_params(axis='both', labelsize=6, width=0.3, length=2)
    scatter = ax.scatter(one_day_grid_features['lon'], one_day_grid_features["lat"],
                         c=one_day_grid_features['inhalation'],
                         s=1.5, ec="k", lw=0, cmap=cmap, norm=norm)

    # Set latitude and longitude labels and ranges
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°N'))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°E'))
    # ax.annotate('N', xy=(0.065, 0.84), xycoords='axes fraction', ha='center',
    #             arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=9)

    # Instead of using LinearLocator, use MaxNLocator and prune='both' to avoid having labels outside of the subplot
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='both'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))

    # Set grid
    ax.grid(which="both", linestyle="--", axis="both", c="lightgray", linewidth=.6, alpha=.3)
    ax.set_axisbelow(True)

    # Set the latitude range
    ax.set_ylim([15, 55])  # This line trims the map to between 20 and 55 degrees north

    # ax.set_aspect(0.9)
    ax.set_title(date, fontsize=9)
    plt.setp(ax.spines.values(), linewidth=0.3)


plt.tight_layout()
fig.subplots_adjust(right=0.91)  # Adjust the right border of the subplots to make room for the colorbar

# Get the position of the first subplot
subplots_pos = axs[3].get_position()

# Define the position and size of the colorbar
cbar_ax = fig.add_axes([0.94, subplots_pos.y0, 0.015, subplots_pos.height + 0.1])

# Add the colorbar
cbar = fig.colorbar(scatter, cax=cbar_ax)

# Change the colorbar's border color and width
for spine in cbar_ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.3)

# Adjust the appearance of the colorbar ticks
cbar.ax.tick_params(labelsize=6, width=0.3, length=2, color='gray')

# Set the number of ticks
cbar.locator = ticker.MaxNLocator(nbins=6)
cbar.update_ticks()

plt.savefig('O3_map.png', format='png', transparent=True, pad_inches=0, dpi=130)
plt.show()