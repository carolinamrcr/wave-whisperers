import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

wave_file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_Ondas\wave_time_series.csv")

wave_data = pd.read_csv(wave_file_path, sep=',', header=0)  # Set header=0 to use the first row as column headers
wave_data['time'] = pd.to_datetime(wave_data['time'])
wave_data.set_index('time', inplace=True)
wave_data['years'] = wave_data.index.year

annual_wave_data9 = wave_data.groupby(wave_data.index.year).quantile(0.99)
annual_wave_data9.index.name = 'years'
annual_wave_data9.drop('years', axis=1, inplace=True)
annual_wave_data5 = wave_data.groupby(wave_data.index.year).quantile(0.5)
annual_wave_data5.index.name = 'years'
annual_wave_data5.drop('years', axis=1, inplace=True)
annual_wave_data1 = wave_data.groupby(wave_data.index.year).quantile(0.1)
annual_wave_data1.index.name = 'years'
annual_wave_data1.drop('years', axis=1, inplace=True)

annual_wave_data = pd.concat([annual_wave_data9, annual_wave_data5, annual_wave_data1], axis=1, keys=['q0.9', 'q0.5', 'q0.1'])

shoreline_file_path = os.path.join(r"C:\Users\danie\Documents\GitHub\wave-whisperers\dataset_linhascosta\CCFT_time_series.csv")

shoreline_data = pd.read_csv(shoreline_file_path)
shoreline_data = shoreline_data.iloc[:, 1:]
shoreline_data['dates'] = pd.to_datetime(shoreline_data['dates'])
shoreline_data.set_index('dates', inplace=True)

annual_shoreline_data = shoreline_data.groupby(shoreline_data.index.year).median(numeric_only=True)
annual_shoreline_data.index.name = 'years'

all_data = pd.concat([annual_wave_data, annual_shoreline_data], axis=1)

# Assuming 'all_data' has a datetime index
all_data_2022 = all_data.loc[[2018, 2019, 2020, 2021, 2022]]

# Calculate the correlation matrix again
correlation_matrix = all_data.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - 2022')
plt.show()


