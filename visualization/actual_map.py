import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import geopandas as gpd
import seaborn as sns
import matplotlib.ticker as ticker

# Use seaborn style defaults and set the default figure size
sns.set(style="whitegrid")

results = pd.read_csv("./dataset/dataset_daily.csv")
# results['date'] = pd.to_datetime(results['date'].astype(str), format='%Y%m%d')

min_value = 0

# 获取 'TMP_P0_L1_GLL0' 这一列的最大值
max_value = 150

# Load country border data
country_borders = gpd.read_file("./visualization/cn-country-border.json")

dates = ['2021-08-16', '2021-08-17', '2021-08-18', '2021-08-19',
         '2021-08-20', '2021-08-21', '2021-08-22', '2021-08-23',
         '2021-08-23', '2021-08-25', '2021-08-26', '2021-08-27'
         ]

# fig, axs = plt.subplots(3, 4, figsize=(10, 5), dpi=130)

# axs = axs.ravel()  # Flatten array of axes
cmap = plt.get_cmap('RdYlBu_r')  # Get the colormap
norm = mcolors.Normalize(vmin=min_value, vmax=max_value)  # Normalize the colormap

for date in dates:
    fig, ax = plt.subplots()

    one_day_grid_features = results[results['date'] == date]
    country_borders.plot(fc="none", ec="gray", ax=ax, lw=0.5, zorder=2)
    # ax.tick_params(axis='both', labelsize=6, width=0.3, length=2)
    scatter = ax.scatter(one_day_grid_features['lon'], one_day_grid_features["lat"],
                         c=one_day_grid_features['o3'],
                         s=5, ec="b", lw=0, cmap=cmap, norm=norm)

    # Set latitude and longitude labels and ranges
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°N'))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°E'))
    # ax.annotate('N', xy=(0.065, 0.84), xycoords='axes fraction', ha='center',
    #             arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=9)

    # Instead of using LinearLocator, use MaxNLocator and prune='both' to avoid having labels outside of the subplot
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='both'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))

    # Set grid
    ax.grid(which="both", linestyle="--", axis="both", c="gray", linewidth=.8, alpha=.3)
    ax.set_axisbelow(True)

    # Set the latitude range
    ax.set_ylim([15, 55])  # This line trims the map to between 20 and 55 degrees north

    # ax.set_aspect(0.9)
    # ax.set_title(date, fontsize=9)
    # plt.setp(ax.spines.values(), linewidth=0.3)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./actual_maps/{date}.png', format='png', transparent=True, pad_inches=0, dpi=130)

#
# plt.tight_layout()
# fig.subplots_adjust(right=0.93)  # Adjust the right border of the subplots to make room for the colorbar
#
# # Get the position of the first subplot
# subplots_pos = axs[11].get_position()

# Define the position and size of the colorbar
# cbar_ax = fig.add_axes([0.95, subplots_pos.y0, 0.01, subplots_pos.height + 0.1])
#
# # Add the colorbar
# cbar = fig.colorbar(scatter, cax=cbar_ax)
#
# # Change the colorbar's border color and width
# for spine in cbar_ax.spines.values():
#     spine.set_edgecolor('gray')
#     spine.set_linewidth(0.3)

# Adjust the appearance of the colorbar ticks
# cbar.ax.tick_params(labelsize=6, width=0.3, length=2, color='gray')
#
# # Set the number of ticks
# cbar.locator = ticker.MaxNLocator(nbins=6)
# cbar.update_ticks()

# plt.savefig('actual_map.png', format='png', transparent=True, pad_inches=0, dpi=130)
# plt.show()
