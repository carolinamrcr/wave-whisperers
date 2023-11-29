import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_Ondas\wave_time_series.csv")

wave_data = pd.read_csv(file_path, sep=',', header=0)  # Set header=0 to use the first row as column headers
wave_data['time'] = pd.to_datetime(wave_data['time'])
wave_data.set_index('time', inplace=True)
wave_data['years'] = wave_data.index.year
wave_data = wave_data[wave_data['years'] != 1983]

def degrees_to_quadrant(wave_direction_degrees):
    if wave_direction_degrees >= 0 and   wave_direction_degrees <= 11.25:
        return 'N'
    elif wave_direction_degrees <= 33.75:
        return 'NNE'
    elif wave_direction_degrees <= 56.25:
        return 'NE'
    elif wave_direction_degrees <= 78.75:
        return 'ENE'
    elif wave_direction_degrees <= 101.25:
        return 'E'
    elif wave_direction_degrees <= 123.75:
        return 'ESE'
    elif wave_direction_degrees <= 146.25:
        return 'SE'
    elif wave_direction_degrees <= 168.75:
        return 'SSE'
    elif wave_direction_degrees <= 191.25:
        return 'S'
    elif wave_direction_degrees <= 213.75:
        return 'SSW'
    elif wave_direction_degrees <= 236.25:
        return 'SW'
    elif wave_direction_degrees <= 258.75:
        return 'WSW'
    elif wave_direction_degrees <= 281.25:
        return 'W'
    elif wave_direction_degrees <= 303.75:
        return 'WNW'
    elif wave_direction_degrees <= 326.25:
        return 'NW'
    elif wave_direction_degrees <= 348.75:
        return 'NNW'
    elif wave_direction_degrees <= 360:
        return 'N'
    else:
        return 'false'

# One-hot encode the 'quadrant_letter' column
wave_data['mwd'] = wave_data['mwd'].apply(degrees_to_quadrant)

one_hot_encode = pd.get_dummies(wave_data['mwd'], prefix='from')

# Concatenate the one-hot encoded columns to the original DataFrame
wave_data = pd.concat([wave_data, one_hot_encode], axis=1)
wave_data = wave_data.drop('mwd', axis=1)

# List of directions
directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

# Iterate through directions and create new columns
for direction in directions:
    # Create new columns for pp1d and swh
    pp1d_column_name = f'pp1d_from_{direction}'
    swh_column_name = f'swh_from_{direction}'
    
    # Use boolean indexing to set values based on the condition
    wave_data[pp1d_column_name] = wave_data['pp1d'] * wave_data[f'from_{direction}']
    wave_data[swh_column_name] = wave_data['swh'] * wave_data[f'from_{direction}']

wave_data.drop(columns=[f'from_{direction}' for direction in directions], inplace=True)
wave_data.drop(columns=['pp1d','swh'], inplace=True)

wave_data = wave_data.replace(0, np.nan)

# Create a new DataFrame for boxplotting
boxplot_data_pp1d = pd.DataFrame()
boxplot_data_swh = pd.DataFrame()

# Populate the new DataFrame with the columns from the original DataFrame
for direction in directions:
    pp1d_column_name = f'pp1d_from_{direction}'
    swh_column_name = f'swh_from_{direction}'
    
    boxplot_data_pp1d[direction] = wave_data[pp1d_column_name]
    boxplot_data_swh[direction] = wave_data[swh_column_name]

#------------------------------------------------------------------------------------------------------------------------------------------------
# Boxplot for pp1d
plt.figure(figsize=(12, 6))
sns.boxplot(data=boxplot_data_pp1d)
plt.title('Boxplot of wave peak period', fontweight='bold')
plt.xlabel('Provenance', fontweight='bold')
plt.ylabel('Peak period (s)', fontweight='bold')
plt.savefig('BP_pp1d.png')  # You can change the file format (e.g., 'your_figure_name.pdf')

#------------------------------------------------------------------------------------------------------------------------------------------------
# Boxplot for swh
plt.figure(figsize=(12, 6))
sns.boxplot(data=boxplot_data_swh)
plt.title('Boxplot of significant wave height', fontweight='bold')
plt.xlabel('Provenance', fontweight='bold')
plt.ylabel('Significant wave height (m)', fontweight='bold')
plt.savefig('BP_swh.png')  # You can change the file format (e.g., 'your_figure_name.pdf')

#------------------------------------------------------------------------------------------------------------------------------------------------

def custom_quantile(series):
    non_zero_values = series[series != 0]
    if len(non_zero_values) > 0:
        return non_zero_values.quantile(0.9)
    else:
        return 0  # You can choose a different default value if needed

# Group by 'years' and apply the custom_quantile function
annual_wave_data = wave_data.groupby('years').agg(custom_quantile).reset_index()
# Set the 'years' column as the index
annual_wave_data.set_index('years', inplace=True)
















