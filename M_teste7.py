import os
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

# Specify the folder path
waves_folder_path = r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_Ondas"
shorelines_folder_path = r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_linhascosta"
transects_folder_path = r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_transects"

# List of file names 
site_names = ['CVCC','CCFT','FTAD','ADLA','LABI',
              'TRAT','ATMC','MCCO','CCCL','NNOR',
              'MEIA','TORR','CVMR','MRMG','MGVR',
              'COSN','VAGR','GBHA','BARR','MIRA']

# Create an empty dictionary to store DataFrames
data = {}

# Loop through each file name
for name in site_names:
    # Construct the file paths
    waves_file_path = os.path.join(waves_folder_path, f"{name}_wave_timeseries.csv")
    shorelines_file_path = os.path.join(shorelines_folder_path, f"{name}_shoreline_timeseries.csv")
    transects_file_path = os.path.join(transects_folder_path, f"{name}_T.geojson")

    # Read the waves CSV files into DataFrame
    waves_df = pd.read_csv(waves_file_path, sep=',', header=0) # Set header=0 to use the first row as column headers
    waves_df['time'] = pd.to_datetime(waves_df['time'])
    waves_df.set_index('time', inplace=True)
    waves_df['years'] = waves_df.index.year
    waves_df['months'] = waves_df.index.month
    waves_df.index = pd.MultiIndex.from_tuples(
    [(year, month) for year, month in zip(waves_df.index.year, waves_df.index.month)],
    names=['years', 'months'])
    waves_df = waves_df[waves_df['years'] != 1983] # Remove 1983 because satellite data is not available for that year
    
    # Drop the 'years and month
    #waves_df = waves_df.drop(['year', 'month'], axis=1)

    # List of directions (16 directions compass rose)
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    def degrees_to_direction(wave_direction_degrees):
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

    # One-hot encode the 'mwd' column
    waves_df['mwd'] = waves_df['mwd'].apply(degrees_to_direction)

    # Create a DataFrame of dummy variables for 'mwd'
    one_hot_encode = pd.get_dummies(waves_df['mwd'], prefix='from')

    # Concatenate the one-hot encoded columns to the original DataFrame
    waves_df = pd.concat([waves_df, one_hot_encode], axis=1)
    waves_df = waves_df.drop('mwd', axis=1)

    # Iterate through directions and create new columns for each direction's pp1d and swh
    for direction in directions:
        # Create new columns for pp1d and swh
        pp1d_column_name = f'pp1d_from_{direction}'
        swh_column_name = f'swh_from_{direction}'
    
        # Use boolean indexing to set values based on the condition
        waves_df[pp1d_column_name] = waves_df['pp1d'] * waves_df[f'from_{direction}']
        waves_df[swh_column_name] = waves_df['swh'] * waves_df[f'from_{direction}']
    
    # Drop the original 'mwd' column and the 'pp1d' and 'swh' columns
    waves_df.drop(columns=[f'from_{direction}' for direction in directions], inplace=True)
    waves_df.drop(columns=['pp1d','swh'], inplace=True)

    # Read the shorelines CSV files into DataFrame
    shorelines_df = pd.read_csv(shorelines_file_path)
    shorelines_df = shorelines_df.iloc[:, 1:]
    shorelines_df['dates'] = pd.to_datetime(shorelines_df['dates'])
    shorelines_df.set_index('dates', inplace=True)
    shorelines_df['years'] = shorelines_df.index.year
    shorelines_df['months'] = shorelines_df.index.month
    shorelines_df.index = pd.MultiIndex.from_tuples(
    [(year, month) for year, month in zip(shorelines_df.index.year, shorelines_df.index.month)],
    names=['years', 'months'])

    # Drop the 'years and month
    #shorelines_df = shorelines_df.drop(['year', 'month'], axis=1)

    # Read the transects GeoJSON file into a GeoDataFrame
    transects_gdf = gpd.read_file(transects_file_path, driver='GeoJSON')

    # Add DataFrames to the dictionary with site name as key
    data[name] = {
        'waves': waves_df,
        'shorelines': shorelines_df,
        'transects': transects_gdf
    }

# Initialize an empty dictionary to store the results
annual_data = {}

# Iterate over keys in the data dictionary
for name in data.keys():
    waves_df = data[name]['waves']
    
    # Group by 'years' and calculate quantiles for each column
    wave_df_annual = waves_df.groupby(level=['years', 'months']).agg(
           {
        'pp1d_from_N': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'swh_from_N': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_NNE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'swh_from_NNE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_NE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'swh_from_NE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_ENE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'swh_from_ENE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_E': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'swh_from_E': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'pp1d_from_ESE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'swh_from_ESE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'pp1d_from_SE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ], 
        'swh_from_SE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'pp1d_from_SSE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'swh_from_SSE': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_S': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'swh_from_S': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'pp1d_from_SSW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'swh_from_SSW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_SW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'swh_from_SW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_WSW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None) 
        ],
        'swh_from_WSW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_W': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'swh_from_W': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_WNW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'swh_from_WNW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_NW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'swh_from_NW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None), 
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None), 
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'pp1d_from_NNW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ],
        'swh_from_NNW': [
            ('10th_quantile', lambda x: x[x != 0].quantile(0.1) if any(x != 0) else None),
            ('50th_quantile', lambda x: x[x != 0].quantile(0.5) if any(x != 0) else None),
            ('90th_quantile', lambda x: x[x != 0].quantile(0.9) if any(x != 0) else None)
        ]})
    
    # Drop the 'years' and 'month' columns
    #wave_df_annual = wave_df_annual.drop(['years', 'month'], axis=1)

    # Replace NaN values with zero
    wave_df_annual = wave_df_annual.fillna(0)

    shoreline_df = data[name]['shorelines']

        # Create a MultiIndex with all possible combinations of years and months
    all_years = shoreline_df.index.get_level_values('years').unique()
    all_months = range(1, 13)
    all_combinations = [(year, month) for year in all_years for month in all_months]

    full_index = pd.MultiIndex.from_tuples(all_combinations, names=['years', 'months'])

    # Group by the MultiIndex and calculate the median
    shoreline_df_annual = shoreline_df.groupby(level=['years', 'months']).median(numeric_only=True)

    # Reindex with the full MultiIndex to fill missing combinations with NaN
    shoreline_df_annual = shoreline_df_annual.reindex(full_index)

    # If needed, you can drop the existing 'months' column
    shoreline_df_annual = shoreline_df_annual.drop('months', axis=1)
    
    # Drop year and month columns
    #shoreline_df_annual = shoreline_df_annual.drop(['years', 'months'], axis=1)

    # Iterate over each column in the DataFrame

    for i in range(1, len(shoreline_df_annual.columns) - 1):
        col = shoreline_df_annual.columns[i]
        prev_col = shoreline_df_annual.columns[i - 1] if i - 1 >= 0 else None
        next_col = shoreline_df_annual.columns[i + 1] if i + 1 < len(shoreline_df_annual.columns) else None

        # Check if there are any NaN values in the current column
        if shoreline_df_annual[col].isnull().any():
            # Fill NaN values with the mean of the available previous and next columns
            if prev_col is not None and next_col is not None:
                shoreline_df_annual[col] = (shoreline_df_annual[prev_col] + shoreline_df_annual[next_col]) / 2
            elif prev_col is not None:
                shoreline_df_annual[col] = shoreline_df_annual[prev_col]
            elif next_col is not None:
                shoreline_df_annual[col] = shoreline_df_annual[next_col]
            else:
                # If there are no immediate previous and next columns, extend the search to 3 columns
                prev_cols = [shoreline_df_annual.columns[j] for j in range(i - 2, i) if j >= 0]
                next_cols = [shoreline_df_annual.columns[j] for j in range(i + 1, i + 4) if j < len(shoreline_df_annual.columns)]

                available_cols = prev_cols + next_cols

                # Filter out None values (columns that are out of range)
                available_cols = [col for col in available_cols if col is not None]

                # Take the mean of available columns
                if len(available_cols) > 0:
                    shoreline_df_annual[col] = shoreline_df_annual[available_cols].mean(axis=1)

    for column in shoreline_df_annual.columns:
        # Check if there are any NaN values in the column
        if shoreline_df_annual[column].isnull().any():
            # Calculate the median value of the column (excluding NaN values)
            median_value = shoreline_df_annual[column].median()
        
            # Replace NaN values with the calculated median value
            shoreline_df_annual[column].fillna(median_value, inplace=True)

    # Add the DataFrame to the dictionary with site name as key
    annual_data[name] = {
        'waves': wave_df_annual,
        'shorelines': shoreline_df_annual
    }





import tensorflow as tf
from tensorflow.keras import layers, Input, Model

# Assuming 'shorelines' and 'waves' have consistent time indices

# Data Preparation
input_names = list(annual_data.keys())
shorelines_data = [annual_data[name]['shorelines'].values for name in input_names]
waves_data = [annual_data[name]['waves'].values for name in input_names]
output_data = [annual_data[name]['shorelines'].values for name in input_names]

# Model Architecture
shoreline_input = Input(shape=(shorelines_data[0].shape[9],), name='shoreline_input')
waves_input = Input(shape=(waves_data[0].shape[1],), name='waves_input')

# Add more layers as needed for your specific problem
combined = layers.concatenate([shoreline_input, waves_input], axis=-1)
hidden_layer = layers.Dense(64, activation='relu')(combined)
output_layer = layers.Dense(shoreline_input.shape[1], activation='linear', name='output')(hidden_layer)

model = Model(inputs=[shoreline_input, waves_input], outputs=output_layer)

# Model Compilation
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Model Training
# Assuming shorelines_data, waves_data, and output_data are numpy arrays
model.fit(x={'shoreline_input': shorelines_data, 'waves_input': waves_data},
          y=output_data,
          epochs=50,
          batch_size=32,
          validation_split=0.2)

# Model Evaluation (optional)
# Evaluate the model on a validation set

# Prediction
# Use the trained model to make predictions on new data







