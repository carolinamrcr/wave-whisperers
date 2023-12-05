import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

wave_file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_Ondas\wave_time_series.csv")

wave_data = pd.read_csv(wave_file_path, sep=',', header=0)  # Set header=0 to use the first row as column headers
wave_data['time'] = pd.to_datetime(wave_data['time'])
wave_data.set_index('time', inplace=True)
wave_data['years'] = wave_data.index.year
wave_data = wave_data[wave_data['years'] != 1983] # Remove 1983 because satellite data is not available for that year

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
wave_data['mwd'] = wave_data['mwd'].apply(degrees_to_direction)

# Create a DataFrame of dummy variables for 'mwd'
one_hot_encode = pd.get_dummies(wave_data['mwd'], prefix='from')

# Concatenate the one-hot encoded columns to the original DataFrame
wave_data = pd.concat([wave_data, one_hot_encode], axis=1)
wave_data = wave_data.drop('mwd', axis=1)

# Iterate through directions and create new columns for each direction's pp1d and swh
for direction in directions:
    # Create new columns for pp1d and swh
    pp1d_column_name = f'pp1d_from_{direction}'
    swh_column_name = f'swh_from_{direction}'
    
    # Use boolean indexing to set values based on the condition
    wave_data[pp1d_column_name] = wave_data['pp1d'] * wave_data[f'from_{direction}']
    wave_data[swh_column_name] = wave_data['swh'] * wave_data[f'from_{direction}']

# Drop the original 'mwd' column and the 'pp1d' and 'swh' columns
wave_data.drop(columns=[f'from_{direction}' for direction in directions], inplace=True)
wave_data.drop(columns=['pp1d','swh'], inplace=True)

# Create a custom function to select the 90th percentile (CAN BE CHANGED)
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
#------------------------------------------------------------------------------------------------------------------------------------------------

shoreline_file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_linhascosta\CCFT_time_series.csv")

shoreline_data = pd.read_csv(shoreline_file_path)
shoreline_data = shoreline_data.iloc[:, 1:]
shoreline_data['dates'] = pd.to_datetime(shoreline_data['dates'])
shoreline_data.set_index('dates', inplace=True)

# Group by 'years' and apply the median function (CAN BE CHANGED) 
annual_shoreline_data = shoreline_data.groupby(shoreline_data.index.year).median(numeric_only=True)

# Set the 'years' column as the index
annual_shoreline_data.index.name = 'years'

# Iterate through the columns to fill NaN values with the mean of adjacent columns
for col in annual_shoreline_data.columns:
    # Check if the column has NaN values
    if annual_shoreline_data[col].isna().any():
        # Find the previous and next columns (if they exist)
        prev_col = annual_shoreline_data.columns[annual_shoreline_data.columns.get_loc(col) - 1] if col != annual_shoreline_data.columns[0] else None
        next_col = annual_shoreline_data.columns[annual_shoreline_data.columns.get_loc(col) + 1] if col != annual_shoreline_data.columns[-1] else None
        
        # Fill NaN values with the mean of adjacent columns
        if prev_col and next_col:
            annual_shoreline_data[col].fillna(annual_shoreline_data[[prev_col, next_col]].mean(axis=1), inplace=True)
        elif prev_col:
            annual_shoreline_data[col].fillna(annual_shoreline_data[prev_col], inplace=True)
        elif next_col:
            annual_shoreline_data[col].fillna(annual_shoreline_data[next_col], inplace=True)

#------------------------------------------------------------------------------------------------------------------------------------------------
# Create training datasets
x_train = annual_wave_data[annual_wave_data.index <= 2014]
y_train = annual_shoreline_data[annual_shoreline_data.index <= 2014]

#Create testing datasets
x_test = annual_wave_data[annual_wave_data.index > 2014]
y_test = annual_shoreline_data[annual_shoreline_data.index > 2014]

# Create an instance of the LinearRegression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# Use the model to predict shoreline positions
y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

bias_matrix = y_test - y_pred
mean_bias = bias_matrix.mean().mean()
print("\nMean bias:", mean_bias)

# Calculate the model performance metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')