import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

wave_file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_Ondas\wave_time_series.csv")

wave_data = pd.read_csv(wave_file_path, sep=',', header=0)  # Set header=0 to use the first row as column headers
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









shoreline_file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_linhascosta\CCFT_time_series.csv")

shoreline_data = pd.read_csv(shoreline_file_path)
shoreline_data = shoreline_data.iloc[:, 1:]
shoreline_data['dates'] = pd.to_datetime(shoreline_data['dates'])
shoreline_data.set_index('dates', inplace=True)

annual_shoreline_data = shoreline_data.groupby(shoreline_data.index.year).median(numeric_only=True)
annual_shoreline_data.index.name = 'years'

all_data = pd.concat([annual_wave_data, annual_shoreline_data], axis=1)

all_data_train = all_data.loc[[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2017, 2018, 2019, 2020, 2021]]

# Assuming 'all_data' has a datetime index
all_data_test = all_data.loc[[2022]]

# Assuming 'all_data' has a datetime index
X_train = all_data_train.drop(columns=['CCFT_1', 'CCFT_2', 'CCFT_3', 'CCFT_4', 'CCFT_5', 'CCFT_6', 'CCFT_7', 'CCFT_8', 'CCFT_9', 'CCFT_10', 'CCFT_11', 'CCFT_12', 'CCFT_13', 'CCFT_14', 'CCFT_15', 'CCFT_16', 'CCFT_17', 'CCFT_18', 'CCFT_19', 'CCFT_20', 'CCFT_21', 'CCFT_22', 'CCFT_23', 'CCFT_24', 'CCFT_25', 'CCFT_26', 'CCFT_27', 'CCFT_28'])
X_train= X_train.replace(np.nan, 0)
y_train = all_data_train[['CCFT_1', 'CCFT_2', 'CCFT_3', 'CCFT_4', 'CCFT_5', 'CCFT_6', 'CCFT_7', 'CCFT_8', 'CCFT_9', 'CCFT_10', 'CCFT_11', 'CCFT_12', 'CCFT_13', 'CCFT_14', 'CCFT_15', 'CCFT_16', 'CCFT_17', 'CCFT_18', 'CCFT_19', 'CCFT_20', 'CCFT_21', 'CCFT_22', 'CCFT_23', 'CCFT_24', 'CCFT_25', 'CCFT_26', 'CCFT_27', 'CCFT_28']]


# Create an instance of the LinearRegression model
model = LinearRegression()

from sklearn.tree import DecisionTreeRegressor
#model = DecisionTreeRegressor()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the shoreline data for the validation years
X_test = all_data_test.drop(columns=['CCFT_1', 'CCFT_2', 'CCFT_3', 'CCFT_4', 'CCFT_5', 'CCFT_6', 'CCFT_7', 'CCFT_8', 'CCFT_9', 'CCFT_10', 'CCFT_11', 'CCFT_12', 'CCFT_13', 'CCFT_14', 'CCFT_15', 'CCFT_16', 'CCFT_17', 'CCFT_18', 'CCFT_19', 'CCFT_20', 'CCFT_21', 'CCFT_22', 'CCFT_23', 'CCFT_24', 'CCFT_25', 'CCFT_26', 'CCFT_27', 'CCFT_28'])
X_test= X_test.replace(np.nan, 0)
y_test = all_data_test[['CCFT_1', 'CCFT_2', 'CCFT_3', 'CCFT_4', 'CCFT_5', 'CCFT_6', 'CCFT_7', 'CCFT_8', 'CCFT_9', 'CCFT_10', 'CCFT_11', 'CCFT_12', 'CCFT_13', 'CCFT_14', 'CCFT_15', 'CCFT_16', 'CCFT_17', 'CCFT_18', 'CCFT_19', 'CCFT_20', 'CCFT_21', 'CCFT_22', 'CCFT_23', 'CCFT_24', 'CCFT_25', 'CCFT_26', 'CCFT_27', 'CCFT_28']]

y_pred = model.predict(X_test)

df_y_pred = pd.DataFrame(y_pred, columns=['CCFT_1', 'CCFT_2', 'CCFT_3', 'CCFT_4', 'CCFT_5', 'CCFT_6', 'CCFT_7', 'CCFT_8', 'CCFT_9', 'CCFT_10', 'CCFT_11', 'CCFT_12', 'CCFT_13', 'CCFT_14', 'CCFT_15', 'CCFT_16', 'CCFT_17', 'CCFT_18', 'CCFT_19', 'CCFT_20', 'CCFT_21', 'CCFT_22', 'CCFT_23', 'CCFT_24', 'CCFT_25', 'CCFT_26', 'CCFT_27', 'CCFT_28'])

print(y_pred)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print(rmse)

from  sklearn.metrics import explained_variance_score

explained_var = explained_variance_score(y_test, y_pred)

print(explained_var)

