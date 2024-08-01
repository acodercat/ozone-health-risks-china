import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import geopandas as gpd
import seaborn as sns
import matplotlib.ticker as ticker

# Use seaborn style defaults and set the default figure size
sns.set(style="whitegrid")

grid_features = pd.read_csv("./dataset/grid_features.csv").sort_values(by='date')

FEATURE_COLUMNS = ['lat', 'lon', 'TMP_P0_L1_GLL0', 'SPFH_P0_2L108_GLL0', 'RH_P0_L4_GLL0', 'PWAT_P0_L200_GLL0',
                   'UGRD_P0_L6_GLL0', 'GUST_P0_L1_GLL0', 'PRES_P0_L7_GLL0', 'CultivatedLand', 'WoodLand',
                   'GrassLand', 'Waters', 'UrbanRural', 'UnusedLand', 'Ocean', 'ELEVATION', 'AOD', 'month', 'weekday']

definition = [{
    "field_name": 'RH_P0_L4_GLL0',
    "name": '相对湿度',
    "unit": '%',
    "cmap": 'viridis'
},
{
    "field_name": 'TMP_P0_L1_GLL0',
    "name": '温度',
    "unit": '\u2103',
    "cmap": 'coolwarm'
},
{
    "field_name": 'PRES_P0_L7_GLL0',
    "name": '大气压',
    "unit": 'Pa',
    "cmap": 'cividis'
},
{
    "field_name": 'UGRD_P0_L6_GLL0',
    "name": '风速',
    "unit": 'm/s',
    "cmap": 'winter'
},
{
    "field_name": 'PWAT_P0_L200_GLL0',
    "name": '可降雨量',
    "unit": 'mm',
    "cmap": 'PuBu'
},
{
    "field_name": 'AOD',
    "name": 'AOD',
    "unit": 'mm',
    "cmap": 'OrRd'
}
]

current_index = 5

cmap = plt.get_cmap(definition[current_index]['cmap'])  # Get the colormap

min_value = grid_features[definition[current_index]['field_name']].min()
max_value = grid_features[definition[current_index]['field_name']].max()

# Load country border data
country_borders = gpd.read_file("./visualization/cn-country-border.json")

dates = ['2020-01-20', '2020-01-25', '2020-01-31']

# fig, axs = plt.subplots(1, 5, figsize=(9, 2), dpi=130)
# fig, axs = plt.subplots(1, 5, figsize=(9, 1.5), dpi=130, sharey=True, sharex=True)

fig, axs = plt.subplots(1, 3, figsize=(8, 2), dpi=130, sharey=True, sharex=True)

# plt.suptitle('Spatial Distribution of Temperature over China', fontsize=14)
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.suptitle(name, fontsize=12)


axs = axs.ravel()  # Flatten array of axes

norm = mcolors.Normalize(vmin=min_value, vmax=max_value)  # Normalize the colormap

for ax, date in zip(axs, dates):
    one_day_grid_features = grid_features[grid_features['date'] == date]
    country_borders.plot(fc="none", ec="gray", ax=ax, lw=0.2, zorder=1.2)
    ax.tick_params(axis='both', labelsize=6, width=0.3, length=2)
    scatter = ax.scatter(one_day_grid_features['lon'], one_day_grid_features["lat"],
                         c=one_day_grid_features[definition[current_index]['field_name']],
                         s=1.5, ec="k", lw=0, cmap=cmap, norm=norm)

    # Set latitude and longitude labels and ranges
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°N'))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°E'))

    ax.set_title(date, fontsize=9)

    # Instead of using LinearLocator, use MaxNLocator and prune='both' to avoid having labels outside of the subplot
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='both'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))

    # Set grid
    ax.grid(which="both", linestyle="--", axis="both", c="lightgray", linewidth=.6, alpha=.3)
    ax.set_axisbelow(True)

    # Set the latitude range
    ax.set_ylim([15, 55])  # This line trims the map to between 20 and 55 degrees north

    # ax.set_aspect(0.9)
    # ax.set_title(date, fontsize=9)
    plt.setp(ax.spines.values(), linewidth=0.3)

plt.tight_layout()
fig.subplots_adjust(right=0.9)  # Adjust the right border of the subplots to make room for the colorbar

# Get the position of the first subplot
subplots_pos = axs[0].get_position()

# Define the position and size of the colorbar
cbar_ax = fig.add_axes([0.91, subplots_pos.y0, 0.01, subplots_pos.height])

# Add the colorbar
cbar = fig.colorbar(scatter, cax=cbar_ax)

# 为色标设置单位标签
cbar.set_label(f'{definition[current_index]["name"]}', rotation=270, labelpad=12,
               fontsize=10)

# Change the colorbar's border color and width
for spine in cbar_ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.3)

# Adjust the appearance of the colorbar ticks
cbar.ax.tick_params(labelsize=6, width=0.6, length=2, color='gray')

# Set the number of ticks
cbar.locator = ticker.MaxNLocator(nbins=6)
cbar.update_ticks()

plt.savefig(f'./dynamic_features_images/{definition[current_index]["name"]}.png', format='png', transparent=True,
            pad_inches=0.1, dpi=130)
plt.show()
