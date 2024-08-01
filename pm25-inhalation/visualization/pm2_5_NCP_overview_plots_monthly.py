import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import geopandas as gpd
import seaborn as sns
import matplotlib.ticker as ticker

# Use seaborn style defaults and set the default figure size
sns.set(style="whitegrid")

results = pd.read_csv("./lgb_NCP_prediction_results.csv")
country_borders = gpd.read_file("./visualization/cn-country-border.json")


results['date'] = pd.to_datetime(results['date'])
results['month'] = results['date'].dt.to_period('M')

results = results.groupby(['grid_id', 'month', 'lat', 'lon']).agg(
    {'pm2_5': 'sum', 'population': 'first'}).reset_index()

specific_provinces = ['北京市', '天津市', '河北省', '山东省', '河南省', '安徽省', '江苏省']
selected_provinces = country_borders[country_borders['name'].isin(specific_provinces)]

months = [
    '2020-01', '2020-02', '2020-03', 
    '2020-04', '2020-05', '2020-06', 
    '2020-07', '2020-08', '2020-09', 
    '2020-10', '2020-11', '2020-12',
]

months = [pd.Period(m) for m in months]
results = results[results['month'].isin(months)]

fig, axs = plt.subplots(3, 4, figsize=(6.6, 6), dpi=130)

axs = axs.ravel()  # Flatten array of axes

colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#800080', '#7E0023']
cmap = mcolors.LinearSegmentedColormap.from_list('pm2_5', colors)
norm = mcolors.Normalize(vmin=0, vmax=results['pm2_5'].max())  # Use Normalize for smooth transition

for ax, month in zip(axs, months):
    one_month_data = results[results['month'] == month]

    ax.tick_params(axis='both', labelsize=6, width=0.3, length=2)
    scatter = ax.scatter(one_month_data['lon'], one_month_data["lat"],
                         c=one_month_data['pm2_5'],
                         s=3, ec="none", lw=0, cmap=cmap, norm=norm)

    # Set latitude and longitude labels and ranges
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°N'))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°E'))

    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))


    # Set grid
    ax.grid(which="both", linestyle="--", axis="both", c="lightgray", linewidth=.6, alpha=.3)
    ax.set_axisbelow(True)

    ax.set_title(month, fontsize=9)
    plt.setp(ax.spines.values(), linewidth=0.3)
    selected_provinces.plot(fc="none", ec="gray", ax=ax, lw=0.2, zorder=1.2)


plt.tight_layout()
fig.subplots_adjust(right=0.95)  # Adjust the right border of the subplots to make room for the colorbar

# Get the position of the first subplot
subplots_pos = axs[11].get_position()

# Define the position and size of the colorbar
cbar_ax = fig.add_axes([0.96, subplots_pos.y0, 0.012, subplots_pos.height + 0.05])

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

plt.savefig('./images/pm2_5_NCP_overview_monthly.png', format='png', transparent=True, dpi=130, bbox_inches='tight')
plt.show()
