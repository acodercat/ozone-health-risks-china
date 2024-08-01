import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import geopandas as gpd
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

# Use seaborn style defaults and set the default figure size
sns.set(style="whitegrid")

results = pd.read_csv("./lgb_prediction_results.csv")
country_borders = gpd.read_file("./visualization/cn-country-border.json")

dates = [
    '2021-01-13', '2021-01-14', '2021-01-15', 
    '2021-01-16', '2021-01-17', '2021-01-18', 
    '2021-01-19', '2021-01-20', '2021-01-21', 
    '2021-01-22', '2021-01-23', '2021-01-24',
]

results = results[results['date'].isin(dates)]
results['inhalation'] = results['population'] * results['pm2_5'] * 0.625

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
vmax = results['inhalation'].quantile(0.96)
norm = mcolors.Normalize(vmin=0, vmax=vmax)

# Iterate over each date and create a separate plot
for date in dates:
    fig, ax = plt.subplots(figsize=(5, 3), dpi=130)
    
    one_day_grid_features = results[results['date'] == date]
    country_borders.plot(fc="none", ec="gray", ax=ax, lw=0, zorder=1.2)
    scatter = ax.scatter(one_day_grid_features['lon'], one_day_grid_features["lat"],
                         c=one_day_grid_features['inhalation'],
                         s=1.5, ec="k", lw=0, cmap=cmap, norm=norm)
    
    

    
    # Set the latitude and longitude range
    ax.set_ylim([15, 55])  # Limit the map to between 15 and 55 degrees north

    # # Set grid lines
    # ax.grid(which="both", linestyle="--", axis="both", c="lightgray", linewidth=.6, alpha=.3)

    # # Remove tick labels (makes labels invisible)
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)

    # Remove the grid
    ax.grid(False)

    ax.axis('off')

    ax.set_aspect('equal')

    # Remove the background by setting the facecolor to 'none'
    ax.set_facecolor('none')

    for spine in ax.spines.values():
        spine.set_visible(False)


    # Adjust the layout to make sure everything fits without overlapping
    plt.tight_layout()
    
    
    # Save each figure with the date in the filename
    plt.savefig(f'./images/pm2_5_inhalation_{date}.png', format='png', transparent=True, dpi=130, bbox_inches='tight')
    
    plt.close(fig)  # Close the figure after saving to free up memory
