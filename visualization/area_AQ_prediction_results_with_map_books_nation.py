from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from shapely.geometry import Point
import matplotlib.ticker as ticker


def generate_figure(area, index, label):
    # Define the file paths
    country_borders_path = "./visualization/cn-country-border.json"

    # Read the country borders and grid static features data
    country_borders = gpd.read_file(country_borders_path)
    prediction_results1 = pd.read_csv(f'./nation3/{area}_1{index}.csv')
    prediction_results2 = pd.read_csv(f'./nation3/{area}_2{index}.csv')
    prediction_results3 = pd.read_csv(f'./nation3/{area}_3{index}.csv')
    static_features = pd.read_csv('./dataset/grid_static_features.csv')

    # Define specific provinces to analyze

    # selected_provinces = country_borders[country_borders['name'].isin(specific_provinces)]

    lon_min, lon_max, lat_min, lat_max = static_features['lon'].min(), static_features['lon'].max(), \
        static_features['lat'].min(), static_features['lat'].max()
    lon = np.arange(lon_min, lon_max + 0.1, 0.5)
    lat = np.arange(lat_min, lat_max + 0.1, 0.5)

    # Create new latitude and longitude grid
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    fig, axs = plt.subplots(1, 3, figsize=(8, 2), dpi=130)

    axs = axs.ravel()  # Flatten array of axes

    colors = [(0, 'green'), (0.2, 'yellow'), (0.4, 'orange'), (0.6, 'red'), (0.8, 'purple'), (1, 'black')]
    cmap_name = 'pm25_cmap'
    pm25_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

    all_prediction_results = [prediction_results1, prediction_results2, prediction_results3]

    # 初始化全局最大和最小值
    global_max = float('-inf')
    global_min = float('inf')

    for prediction_result in all_prediction_results:
        local_max = prediction_result['pred'].max()
        local_min = prediction_result['pred'].min()
        if local_max > global_max:
            global_max = local_max

        if local_min < global_min:
            global_min = local_min

    min_value = global_min
    max_value = global_max

    norm = mcolors.Normalize(vmin=min_value,
                             vmax=max_value)

    for ax, prediction_results in zip(axs, all_prediction_results):
        prediction_results = pd.merge(prediction_results, static_features, on='grid_id', how='left')
        # Linear interpolation
        value_linear = griddata((prediction_results['lon'], prediction_results['lat']), prediction_results['pred'],
                                (lon_grid, lat_grid), method='linear')
        # Convert to DataFrame
        df_interpolated = pd.DataFrame(
            {'lon': lon_grid.ravel(), 'lat': lat_grid.ravel(), 'value': value_linear.ravel()})

        # Create GeoDataFrame and set CRS
        geometry = [Point(xy) for xy in zip(df_interpolated['lon'], df_interpolated['lat'])]
        gdf_interpolated = gpd.GeoDataFrame(df_interpolated, geometry=geometry, crs="EPSG:4326")

        # Clip data with China borders
        gdf_interpolated_clipped = gpd.overlay(gdf_interpolated, country_borders, how='intersection')

        ax.tick_params(axis='both', labelsize=6, width=0.3, length=2)

        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°N'))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1d°E'))

        # Plot interpolated elevation values for specific provinces
        scatter = ax.scatter(gdf_interpolated_clipped['lon'], gdf_interpolated_clipped['lat'],
                             c=gdf_interpolated_clipped['value'],
                             s=7, cmap=pm25_cmap, norm=norm, marker='s')

        # Set the latitude range
        ax.set_ylim([15, 55])  # This line trims the map to between 20 and 55 degrees north

        # Adjust axis tick positions
        # ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune='both'))
        # ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='both'))

        # Set grid
        ax.grid(which="both", linestyle="--", axis="both", c="lightgray", linewidth=.6, alpha=.3)
        ax.set_axisbelow(True)
        plt.setp(ax.spines.values(), linewidth=0.3)

        # Plot the borders of the selected provinces
        country_borders.plot(fc="none", ec="gray", ax=ax, lw=0.5, zorder=1.2)

    # sm = plt.cm.ScalarMappable(cmap=pm25_cmap, norm=norm)
    #
    # pos = axs[0].get_position()
    # cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
    # fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

    # Get the position of the first subplot
    subplots_pos = axs[0].get_position()

    # Define the position and size of the colorbar
    cbar_ax = fig.add_axes([0.91, subplots_pos.y0, 0.01, subplots_pos.height])

    # Add the colorbar
    cbar = fig.colorbar(scatter, cax=cbar_ax)

    # Change the colorbar's border color and width
    for spine in cbar_ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.3)

    # Adjust the appearance of the colorbar ticks
    cbar.ax.tick_params(labelsize=6, width=0.6, length=2, color='gray')

    # Set the number of ticks
    cbar.locator = ticker.MaxNLocator(nbins=6)
    cbar.update_ticks()

    cbar.set_label(label, rotation=270, labelpad=12,
                   fontsize=8)

    # Save figure
    plt.savefig(f'./第五章-图3/{area}_{index}_scatter.png', format='png', transparent=True, pad_inches=0.1, dpi=130,
                bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    area = '全国'
    definitions = [
        {
            "index": 'so2',
            "label": 'SO2 (µg/m³)'
        },
        {
            "index": 'pm10',
            "label": 'PM10 (µg/m³)'
        },
        {
            "index": 'pm2_5',
            "label": 'PM2.5 (µg/m³)'
        },
        {
            "index": 'o3',
            "label": 'O3 (µg/m³)'
        },
        {
            "index": 'co',
            "label": 'CO (µg/m³)'
        },
        {
            "index": 'no2',
            "label": 'NO2 (µg/m³)'
        }
    ]
    for definition in definitions:
        generate_figure(area, definition['index'], definition['label'])
        # break
