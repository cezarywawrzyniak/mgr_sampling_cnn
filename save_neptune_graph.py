import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Read the CSV file into a pandas DataFrame
csv_file = 'architectures/training_val_loss_2D.csv'  # Replace with the actual path to your CSV file
data = pd.read_csv(csv_file, header=None)

# Convert the Unix timestamps to datetime objects (assuming timestamps are in the second column)
data['timestamp'] = pd.to_datetime(data.iloc[:, 1], unit='ms')

# Calculate relative time from the start of training in hours
data['relative_time_hours'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds() / 3600

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(data['relative_time_hours'], data.iloc[:, 2], marker='o')  # Assuming val_loss is in the third column
plt.xlabel('Czas względny (godziny)')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Podczas Treningu')
plt.grid(True)

# Add labels to each data point based on the first column in data
for i, row in data.iterrows():
    label = str(int(row.iloc[0]))  # Convert the label to an integer and then to a string
    plt.text(row['relative_time_hours'], row.iloc[2], label, fontsize=7, ha='left', va='bottom')

# Add annotation to indicate that text on the data points is the epoch number
plt.text(0.93, 0.16, "Numery epok", transform=plt.gca().transAxes, ha='right', va='top', fontsize=10, color='gray')

# Save the plot as an SVG file
output_file = 'architectures/validation_loss_plot.svg'
plt.savefig(output_file, format='svg')

# Show the plot (optional)
# plt.show()
