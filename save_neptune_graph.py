import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'architectures/training_val_loss_3D.csv'
data = pd.read_csv(csv_file, header=None)

data['timestamp'] = pd.to_datetime(data.iloc[:, 1], unit='ms')

data['relative_time_hours'] = (data['timestamp'] - data['timestamp'].min()).dt.total_seconds() / 3600

plt.figure(figsize=(10, 6))
plt.plot(data['relative_time_hours'], data.iloc[:, 2], marker='o', color='r')
plt.xlabel('Czas względny (godziny)')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Podczas Treningu')
plt.grid(True)

for i, row in data.iterrows():
    label = str(int(row.iloc[0]))
    plt.text(row['relative_time_hours'], row.iloc[2], label, fontsize=7, ha='left', va='bottom')

plt.text(0.93, 0.16, "Numery epok", transform=plt.gca().transAxes, ha='right', va='top', fontsize=10, color='gray')

output_file = 'architectures/validation_loss_plot_3D.svg'
plt.savefig(output_file, format='svg')

# plt.show()
