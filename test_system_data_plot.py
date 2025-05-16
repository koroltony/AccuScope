import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib.dates as mdates

# Step 1: Read the data from the file
with open('arthrex_system_logs.txt', 'r') as file:
    data = file.readlines()

# Step 2: Parse the data into a pandas DataFrame
records = [json.loads(line) for line in data]
df = pd.DataFrame(records)

# Convert time to datetime
df['time'] = pd.to_datetime(df['time'], format='%m-%d-%Y %H:%M:%S.%f')

# Convert uptime to hours (float)
df['uptime'] = df['uptime'].str.extract(r'(\d+\.\d+)').astype(float)

# Check the data types of the columns
# print(df.dtypes)

# Convert percentage strings to floats
for column in df.columns:
    if column not in ['time', 'uptime'] and df[column].dtype == 'object':
        df[column] = df[column].str.rstrip('%').astype(float)

# Step 3: Create the visualization
fig, axs = plt.subplots(4, 1, figsize=(15, 15), sharex=True)

# Plot CPU usage
for i, cpu in enumerate(['cpu1', 'cpu2', 'cpu3', 'cpu4', 'cpu6', 'cpu7', 'cpu8'], start=1):
    axs[0].plot(df['time'], df[cpu], label=f'CPU{i}')
axs[0].set_title('CPU Usage')
axs[0].set_ylabel('Usage (%)')
axs[0].legend()

# Plot GPU usage
axs[1].plot(df['time'], df['gpu'], label='GPU', color='orange')
axs[1].set_title('GPU Usage')
axs[1].set_ylabel('Usage (%)')
axs[1].legend()

# Plot RAM usage
axs[2].plot(df['time'], df['ram'], label='RAM', color='green')
axs[2].set_title('RAM Usage')
axs[2].set_ylabel('Usage (%)')
axs[2].legend()

# Plot Swap usage
axs[3].plot(df['time'], df['swap'], label='Swap', color='red')
axs[3].set_title('Swap Usage')
axs[3].set_ylabel('Usage (%)')
axs[3].legend()

# Set x-axis label
axs[3].set_xlabel('Time')

# Format x-axis labels for better readability
date_format = mdates.DateFormatter('%m-%d %H:%M')
axs[3].xaxis.set_major_formatter(date_format)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()
