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

# Convert units (%, C, W, RPM) to numeric
for column in df.columns:
    if column not in ['time', 'uptime'] and df[column].dtype == 'object':
        df[column] = df[column].str.replace('%', '', regex=False)\
                               .str.replace('W', '', regex=False)\
                               .str.replace('C', '', regex=False)\
                               .str.replace('RPM', '', regex=False)
        df[column] = pd.to_numeric(df[column], errors='coerce')

# Step 3: Create the visualization with 7 subplots
fig, axs = plt.subplots(7, 1, figsize=(15, 20), sharex=True)

# 1. CPU usage
for i, cpu in enumerate(['cpu1', 'cpu2', 'cpu3', 'cpu4', 'cpu6', 'cpu7', 'cpu8'], start=1):
    axs[0].plot(df['time'], df[cpu], label=f'CPU{i}')
axs[0].set_title('CPU Usage')
axs[0].set_ylabel('Usage (%)')
axs[0].legend()

# 2. GPU usage
axs[1].plot(df['time'], df['gpu'], label='GPU', color='orange')
axs[1].set_title('GPU Usage')
axs[1].set_ylabel('Usage (%)')
axs[1].legend()

# 3. RAM usage
axs[2].plot(df['time'], df['ram'], label='RAM', color='green')
axs[2].set_title('RAM Usage')
axs[2].set_ylabel('Usage (%)')
axs[2].legend()

# 4. Swap usage
axs[3].plot(df['time'], df['swap'], label='Swap', color='red')
axs[3].set_title('Swap Usage')
axs[3].set_ylabel('Usage (%)')
axs[3].legend()

# 5. Temperatures
for temp_col, label in zip(['Temp CPU', 'Temp GPU', 'Temp Thermal'], ['CPU', 'GPU', 'Thermal']):
    axs[4].plot(df['time'], df[temp_col], label=f'{label} Temp')
axs[4].set_title('Temperatures')
axs[4].set_ylabel('°C')
axs[4].legend()

# 6. Fan speed
axs[5].plot(df['time'], df['Fan Speed'], label='Fan Speed', color='purple')
axs[5].set_title('Fan Speed')
axs[5].set_ylabel('RPM')
axs[5].legend()

# 7. Power consumption
axs[6].plot(df['time'], df['CPU Power'], label='CPU Power', color='brown')
axs[6].plot(df['time'], df['GPU Power'], label='GPU Power', color='cyan')
axs[6].set_title('Power Consumption')
axs[6].set_ylabel('Watts')
axs[6].set_xlabel('Time')
axs[6].legend()

# Format x-axis for time
date_format = mdates.DateFormatter('%m-%d %H:%M')
axs[6].xaxis.set_major_formatter(date_format)
plt.xticks(rotation=45)

# Layout
plt.tight_layout()
plt.show()
