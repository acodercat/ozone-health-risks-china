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
results['date'] = pd.to_datetime(results['date'])
results['month'] = results['date'].dt.to_period('M')

results = results.groupby(['grid_id', 'month', 'lat', 'lon']).agg(
    {'pm2_5': 'sum', 'population': 'first'}).reset_index()

country_borders = gpd.read_file("./visualization/cn-country-border.json")

months = [
    '2020-01', '2020-02', '2020-03', 
    '2020-04', '2020-05', '2020-06', 
    '2020-07', '2020-08', '2020-09', 
    '2020-10', '2020-11', '2020-12',
]

months = [pd.Period(m) for m in months]
results = results[results['month'].isin(months)]

colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#800080', '#7E0023']
cmap = mcolors.LinearSegmentedColormap.from_list('pm2_5', colors)
norm = mcolors.Normalize(vmin=0, vmax=results['pm2_5'].max())

for month in months:
    fig, ax = plt.subplots(figsize=(5, 3), dpi=130)
    one_month_data = results[results['month'] == month]
    
    country_borders.plot(fc="none", ec="gray", ax=ax, lw=0.1, zorder=1)
    scatter = ax.scatter(one_month_data['lon'], one_month_data['lat'],
                         c=one_month_data['pm2_5'],
                         s=1.5, ec="k", lw=0, cmap=cmap, norm=norm)
 
    ax.set_ylim([15, 55])
    ax.tick_params(axis='both', which='major', labelsize=8, color='gray')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f°N'))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f°E'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='both'))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
    
    ax.grid(which="both", linestyle="--", axis="both", c="lightgray", linewidth=.6, alpha=.3)
    ax.set_axisbelow(True)
    

    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    
    plt.tight_layout()
    
    # Add colorbar to the single subplot
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6, width=0.3, length=2, color='gray', which='both')

    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.3)
    
    # Save the figure
    fig.savefig(f'./images/pm2_5_{month}.png', format='png', transparent=True, dpi=130, bbox_inches='tight')
    
    plt.close(fig)  # Close the figure to avoid display
