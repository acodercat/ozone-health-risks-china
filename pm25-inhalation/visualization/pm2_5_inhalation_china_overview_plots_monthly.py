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


# 将date列转换为datetime对象
results['date'] = pd.to_datetime(results['date'])

# 添加一个月份列用于分组
results['month'] = results['date'].dt.to_period('M')

# 按照grid_id和月份分组，对pm2_5求和，并保持其他列
results = results.groupby(['grid_id', 'month', 'lat', 'lon']).agg(
    {'pm2_5': 'sum', 'population': 'first'}).reset_index()



# Load country border data
country_borders = gpd.read_file("./visualization/cn-country-border.json")

months = [
            '2020-01', '2020-02', '2020-03', 
            '2020-04', '2020-05', '2020-06', 
            '2020-07', '2020-08', '2020-09', 
            '2020-10', '2020-11', '2020-12',
        ]


months = [pd.Period(m) for m in months]
results = results[results['month'].isin(months)]

results['inhalation'] = results['population'] * results['pm2_5'] * 0.625

fig, axs = plt.subplots(3, 4, figsize=(10, 5), dpi=130)
# plt.suptitle('Spatial Distribution of Ozone over China', fontsize=14)

axs = axs.ravel()  # Flatten array of axes

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

# 创建一个线性渐变的自定义colormap
cmap_name = 'my_custom_cmap'
cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=2000)

vmax = results['inhalation'].quantile(0.98)
norm = mcolors.Normalize(vmin=0, vmax=vmax)

for ax, month in zip(axs, months):
    one_month_grid_results = results[results['month'] == month]
    country_borders.plot(fc="none", ec="gray", ax=ax, lw=0.2, zorder=1.2)
    ax.tick_params(axis='both', labelsize=6, width=0.3, length=2)
    scatter = ax.scatter(one_month_grid_results['lon'], one_month_grid_results["lat"],
                         c=one_month_grid_results['inhalation'],
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

    ax.set_title(month, fontsize=9)
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

plt.savefig('./images/pm2_5_inhalation_overview_monthly.png', format='png', transparent=True, dpi=130, bbox_inches='tight')
plt.show()
