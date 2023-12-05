import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import contextily as ctx

# Load transect GeoJSON file into a GeoDataFrame
transect_file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\CoastSat\examples\CCFT_transects.geojson")
transects_gdf = gpd.read_file(transect_file_path, driver='GeoJSON')

# Create a dictionary to store transect coordinates
transects = dict([])
for i in transects_gdf.index:
    transect_name = transects_gdf.loc[i, 'name']
    start_point = np.array(transects_gdf.loc[i, 'geometry'].coords[0])
    end_point = np.array(transects_gdf.loc[i, 'geometry'].coords[-1])
    transects[transect_name] = {'start': start_point, 'end': end_point}

# Set up the plot figure
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlabel('Eastings')
ax.set_ylabel('Northings')
ax.grid(linestyle=':', color='0.5')

# Plot transects
for key, coordinates in transects.items():
    ax.plot(coordinates['start'][0], coordinates['start'][1], 'bo', ms=5)
    #ax.plot(coordinates['end'][0], coordinates['end'][1], 'bo', ms=5)
    ax.plot([coordinates['start'][0], coordinates['end'][0]],
            [coordinates['start'][1], coordinates['end'][1]], 'k-', lw=1)

# Add OSM (OpenStreetMap) as background
ctx.add_basemap(ax, crs=transects_gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)

# Load shoreline data
shoreline_file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_linhascosta\CCFT_shoreline_timeseries.csv")
shoreline_data = pd.read_csv(shoreline_file_path)
shoreline_data['dates'] = pd.to_datetime(shoreline_data['dates'])
shoreline_data.set_index('dates', inplace=True)

# Plot connected shoreline points for each year
for year in shoreline_data.index.year.unique():
    combined_points = []
    for transect_name, coordinates in transects.items():
        shoreline_value = shoreline_data.loc[shoreline_data.index.year == year, transect_name].values[0]
        direction_vector = (coordinates['end'] - coordinates['start']) / np.linalg.norm(coordinates['end'] - coordinates['start'])
        intersection_point = coordinates['start'] + shoreline_value * direction_vector
        combined_points.append(intersection_point)
    combined_points = np.array(combined_points)
    ax.plot(combined_points[:, 0], combined_points[:, 1], label=str(year))

# Customize your plot (labels, legend, etc.)
ax.set_title('Connected Shoreline Points with Transects Over Time')
#ax.legend()

# Show the plot
plt.show()







