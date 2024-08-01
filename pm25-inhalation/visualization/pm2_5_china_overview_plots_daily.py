import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import geopandas as gpd
import seaborn as sns
import matplotlib.ticker as ticker

# Use seaborn style defaults and set the default figure size
sns.set(style="whitegrid")

results = pd.read_csv("./lgb_prediction_results.csv")
# results['date'] = pd.to_datetime(results['date'].astype(str), format='%Y%m%d')



# Load country border data
country_borders = gpd.read_file("./visualization/cn-country-border.json")

dates = [
            '2021-01-13', '2021-01-14', '2021-01-15', 
            '2021-01-16', '2021-01-17', '2021-01-18', 
            '2021-01-19', '2021-01-20', '2021-01-21', 
            '2021-01-22', '2021-01-23', '2021-01-24',
        ]


results = results[results['date'].isin(dates)]


min_value = results['pm2_5'].min()
# max_value = 200


fig, axs = plt.subplots(3, 4, figsize=(10, 5), dpi=130)
# plt.suptitle('Spatial Distribution of Ozone over China', fontsize=14)

axs = axs.ravel()  # Flatten array of axes

colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#800080', '#7E0023']
cmap = mcolors.LinearSegmentedColormap.from_list('pm2_5', colors)
norm = mcolors.Normalize(vmin=0, vmax=250)  # Use Normalize for smooth transition

for ax, date in zip(axs, dates):
    one_day_grid_features = results[results['date'] == date]
    country_borders.plot(fc="none", ec="gray", ax=ax, lw=0.2, zorder=1.2)
    ax.tick_params(axis='both', labelsize=6, width=0.3, length=2)
    scatter = ax.scatter(one_day_grid_features['lon'], one_day_grid_features["lat"],
                         c=one_day_grid_features['pm2_5'],
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

    ax.set_title(date, fontsize=9)
    plt.setp(ax.spines.values(), linewidth=0.3)


plt.tight_layout()
fig.subplots_adjust(right=0.95)  # Adjust the right border of the subplots to make room for the colorbar

# Get the position of the first subplot
subplots_pos = axs[11].get_position()

# Define the position and size of the colorbar
cbar_ax = fig.add_axes([0.96, subplots_pos.y0, 0.01, subplots_pos.height + 0.1])

# Add the colorbar
cbar = fig.colorbar(scatter, cax=cbar_ax)

# Change the colorbar's border color and width
for spine in cbar_ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.3)

cbar.ax.tick_params(labelsize=6, width=0.3, length=2, color='gray', which='both')
cbar.ax.yaxis.set_ticks_position('right')
cbar.locator = ticker.MaxNLocator(nbins=6)
cbar.update_ticks()

plt.savefig('./images/pm2_5_china_overview_daily.png', format='png', transparent=True, dpi=130, bbox_inches='tight')
plt.show()
