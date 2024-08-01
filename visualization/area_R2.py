import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

# Define the file paths
country_borders_path = "./visualization/cn-country-border.json"
prediction_results_path = './dataset/YRD_grid_R2.csv'

# Read the country borders and grid static features data
country_borders = gpd.read_file(country_borders_path)
prediction_results = pd.read_csv(prediction_results_path)

# Define specific provinces to analyze
specific_provinces = ['北京市', '天津市', '河北省', '山东省', '河南省', '安徽省', '江苏省']
# specific_provinces = ['上海市', '江苏省', '浙江省', '安徽省']
selected_provinces = country_borders[country_borders['name'].isin(specific_provinces)]

print(prediction_results['r2'].mean())

# Create figure and axes for plotting
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°N'))
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°E'))

# Define colormap
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=prediction_results['r2'].min(),
                         vmax=prediction_results['r2'].max())

# Plot interpolated elevation values for specific provinces
ax.scatter(prediction_results['lon'], prediction_results['lat'], c=prediction_results['r2'],
           s=5, cmap=cmap, norm=norm)

# Adjust axis tick positions
ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=8, prune='both'))
ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune='both'))

# Plot the borders of the selected provinces
selected_provinces.plot(fc="none", ec="gray", ax=ax, lw=0.5, zorder=1.2)


sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

pos = ax.get_position()
cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

# Save figure
# plt.savefig('NCP_R2_scatter.png', format='png', transparent=True, pad_inches=0, dpi=130, bbox_inches='tight')
plt.show()
plt.close()
