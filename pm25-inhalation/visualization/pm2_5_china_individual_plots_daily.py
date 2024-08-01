import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
import matplotlib.colors as mcolors

# Set seaborn style defaults and set the default figure size
sns.set(style="whitegrid")

# Read in the prediction results
results = pd.read_csv("./lgb_prediction_results.csv")

# Load country border data
country_borders = gpd.read_file("./visualization/cn-country-border.json")

# Define the dates for which we want to visualize the data
dates = [
    '2021-01-13', '2021-01-14', '2021-01-15', 
    '2021-01-16', '2021-01-17', '2021-01-18', 
    '2021-01-19', '2021-01-20', '2021-01-21', 
    '2021-01-22', '2021-01-23', '2021-01-24',
]

# Filter the results for the specified dates
results = results[results['date'].isin(dates)]

# Define the color map for the PM2.5 values using the colors you provided
colors = ['#00E400', '#FFFF00', '#FF7E00', '#FF0000', '#800080', '#7E0023']
cmap = mcolors.LinearSegmentedColormap.from_list('pm2_5', colors)

# Create a normalized color scale for the PM2.5 values
norm = mcolors.Normalize(vmin=0, vmax=250)

# Loop through each date and create a scatter plot
for date in dates:
    fig, ax = plt.subplots(figsize=(5, 3), dpi=130)

    # Filter the data for the current day
    one_day_grid_features = results[results['date'] == date]
    
    # Plot country borders
    country_borders.plot(fc="none", ec="gray", ax=ax, lw=0, zorder=1.2)
    
    # Create a scatter plot of PM2.5 values
    scatter = ax.scatter(
        one_day_grid_features['lon'], one_day_grid_features["lat"],
        c=one_day_grid_features['pm2_5'],
        s=1.5, ec="k", lw=0, cmap=cmap, norm=norm
    )

    # Set the latitude and longitude range
    ax.set_ylim([15, 55])  # Limit the map to between 15 and 55 degrees north

    # # Set grid lines
    # ax.grid(which="both", linestyle="--", axis="both", c="lightgray", linewidth=.6, alpha=.3)

    # # Remove tick labels (makes labels invisible)
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)


    ax.axis('off')

    ax.set_aspect('equal')

    # Remove the grid
    ax.grid(False)

    # Remove the background by setting the facecolor to 'none'
    ax.set_facecolor('none')

    for spine in ax.spines.values():
        spine.set_visible(False)


    # Adjust the layout to make sure everything fits without overlapping
    plt.tight_layout()
    
    # Save the figure with the date in the filename
    plt.savefig(f'./images/pm2_5_{date}.png', format='png', transparent=True, dpi=130, bbox_inches='tight')
    
    # Close the figure to free up memory
    plt.close(fig)
