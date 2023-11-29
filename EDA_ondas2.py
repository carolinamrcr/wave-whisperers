

#------------------------------------------------------------------------------------------------------------------------------------------------
# Create a figure with subplots arranged horizontally
fig, axs = plt.subplots(1, len(wave_data.columns)-1, figsize=(15, 5))

# Plot each boxplot in a separate subplot
columns_to_plot = wave_data.columns.drop('years')
for i, column in enumerate(columns_to_plot):
    axs[i].boxplot(wave_data[column])
    axs[i].set_title(column)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------
# Output directory to save the figures
output_dir = 'output_figures'
os.makedirs(output_dir, exist_ok=True)

# Define columns
mwd_column = 'mwd'  # mean wave direction
pp1d_column = 'pp1d'  # peak period
swh_column = 'swh'  # significant wave height

# Group data by 'years' and create boxplots for each year
grouped_data = wave_data.groupby('years')

# Iterate over each year and save boxplots as PNG files
for year, data in grouped_data:
    # Create a figure with subplots arranged horizontally
    fig, axs = plt.subplots(1, len(data.columns) - 1, figsize=(18, 5))  # Adjust the figure size

    # Plot each boxplot in a separate subplot
    columns_to_plot = data.columns.drop('years')
    for i, column in enumerate(columns_to_plot):
        axs[i].boxplot(data[column])
        axs[i].set_title('')  # Clear the default title
        
        # Set y-axis label based on the column
        if column == mwd_column:
            axs[i].set_ylabel('Mean Wave Direction (°)', fontweight='bold')
        elif column == pp1d_column:
            axs[i].set_ylabel('Peak Period (s)', fontweight='bold')
        elif column == swh_column:
            axs[i].set_ylabel('Significant Wave Height (m)', fontweight='bold')

    # Adjust layout and add title
    plt.suptitle(f'Boxplots for {year}', fontweight='bold')

    # Manually adjust layout
    plt.subplots_adjust(wspace=0.4)  # Adjust the horizontal space between subplots
    
    # Save the figure as a PNG file
    output_file = os.path.join(output_dir, f'BP_{year}.png')
    plt.savefig(output_file, bbox_inches='tight')
    
    # Close the figure to release resources
    plt.close()

# Display a message when all figures are saved
print(f"All figures saved in '{output_dir}' directory.")