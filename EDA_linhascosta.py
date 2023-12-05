import os
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy

# Specify the location code
location_code = 'MEIA'

# Define the location mapping
location_mapping = {
    'CCFT': 'Costa da Caparica to Fonte da Telha',
    'MEIA': 'Meia Praia (Lagos)',
    'VAGR': 'Vagueira',
    'TROI': 'Tróia',
    'NNOR': 'Nazaré Norte',
}

# Use the mapping to get the corresponding location name
location_name = location_mapping.get(location_code, location_code)

# Assuming the files are in the same folder and follow a naming pattern
file_name = f"{location_code}_time_series.csv"
file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_linhascosta", file_name)

# Check if the file exists before reading
if os.path.exists(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    data = data.iloc[:, 1:]
    data['dates'] = pd.to_datetime(data['dates'])
    data.set_index('dates', inplace=True)
    for col in data.columns.tolist():
        new_col_name = 'chan' + col
        data[new_col_name] = data[col].diff()
        data = data.drop(col, axis=1)
else:
    print(f"File not found for location code '{location_code}'.")

print(data.head())

# Calculate the annual median for numeric values of the same year
annual_data = data.groupby(data.index.year).sum(numeric_only=True)
annual_data.index.name = 'years'

#------------------------------------------------------------------------------------------------------------------------------------------------
# Create figure for evolution of the median values over the years
fig, ax = plt.subplots()

# Plot each column values over the years
annual_data.plot(grid=True, ax=ax)

# Set the title
ax.set_title(f'Evolution for {location_name} (Annual Median)', fontweight='bold')

# Set the grid
ax.grid(linestyle="--")

# Set the labels
ax.set_ylabel('Distance to transect head (m)', fontweight='bold')
ax.set_xlabel('Years', fontweight='bold')

# Set the x-axis limits
ax.set_xlim(annual_data.index.min(), annual_data.index.max())

# Set the x-axis ticks to be yearly
ax.set_xticks(annual_data.index)
ax.tick_params(axis='x', rotation=45)

# Add the legend at the bottom with 30 columns
ax.legend(ncol=14, loc='lower center', bbox_to_anchor=(0.5, -0.25))

# Adjust subplot parameters to compress the plot box
fig.subplots_adjust(bottom=0.25)  # You can adjust the 'bottom' parameter

# Show the plot
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------
# Create a figure for the boxplot
fig, ax = plt.subplots()

# Plot boxplot for every column of annual_data
annual_data.boxplot(ax=ax)

# Set the x-axis ticks and their labels
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontweight='bold')

# Set the title and labels
ax.set_title(f'Boxplot for {location_name} (Annual Median)', fontweight='bold')
ax.set_ylabel('Distance to transect head (m)', fontweight='bold')

# Set the grid
ax.grid(linestyle="--")

# Show the plot
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate the correlation matrix
correlation_matrix = annual_data.corr()

# Create a figure for the correlation matrix
fig, ax = plt.subplots()

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax,
            annot_kws={"size": 8})  # Adjust the size of annotations as needed


# Set the title
ax.set_title(f'Correlation matrix for {location_name} (Annual Median)', fontweight='bold')

# Set ticks to appear on all sides and position them in the middle of the cells
ax.set_xticks([i + 0.5 for i in range(len(correlation_matrix.columns))])
ax.set_yticks([i + 0.5 for i in range(len(correlation_matrix.index))])

# Make ticks bold
ax.set_xticklabels(correlation_matrix.columns, fontweight='bold')
ax.set_yticklabels(correlation_matrix.index, fontweight='bold')

# Show the plot
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------
# Compute the distance matrix
annual_data = annual_data.apply(lambda col: col.fillna(col.median()))
annual_data = annual_data.dropna(axis=1, how='all')
distance_matrix = annual_data.corr().values

# Perform hierarchical clustering
linkage_matrix = hierarchy.linkage(distance_matrix, method='average')

# Create a larger figure
fig, ax = plt.subplots()

# Plot the dendrogram inside the subplot
dendrogram = hierarchy.dendrogram(linkage_matrix, labels=annual_data.columns, ax=ax)
ax.set_title(f'Dendogram for clustering the transects at {location_name} (Annual Median)', fontweight='bold')

ax.set_ylabel('Distance of the clusters', fontweight='bold')
ax.set_xticklabels(annual_data.columns, fontweight='bold')
ax.tick_params(axis='x', rotation=90)

# Set the grid
ax.grid(linestyle="--")

# Show the entire figure
plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------



