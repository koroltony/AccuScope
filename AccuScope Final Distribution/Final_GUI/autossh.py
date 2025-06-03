from fabric import Connection, Config
import getpass
import time
import sys
import signal
import shutil
from datetime import datetime, timedelta
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

highlight_window = timedelta(seconds=0.5)

sys.stdout.reconfigure(line_buffering=True)  # <-- Force print to flush immediately

if len(sys.argv) != 5:
    print("Usage: python autossh.py <ip> <username> <password> <case_system_log_path>")
    sys.exit(1)

ip = sys.argv[1]
username = sys.argv[2]
password = sys.argv[3]
case_log_path = sys.argv[4]

conn = Connection(
    host=ip,
    user=username,
    connect_kwargs={'password': password}
)

print("Starting remote monitoring script...")
conn.run('nohup python3 jtopTestScriptAdditions.py > system_logs.txt 2>&1 &', pty=False)
print("Remote script running in background.")

    
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define shutdown logic
def cleanup_and_exit(signum=None, frame=None):
    print("Shutting down remote monitoring...")
    try:
        conn.run("pkill -f jtopTestScriptAdditions.py")
        print("Remote script killed.")

        remote_path = './system_logs.txt'
        local_path = os.path.join(script_dir, "arthrex_system_logs.txt")
        conn.get(remote_path, local_path)
        print(f'Downloaded {remote_path} to {local_path}')

        # Append remote log to the case log file
        with open(case_log_path, 'a') as case_log_file, open(local_path, 'r') as remote_log_file:
            case_log_file.write("\n--- Remote Monitoring Output ---\n")
            shutil.copyfileobj(remote_log_file, case_log_file)
        print(f"Appended remote logs to {case_log_path}")

        # Define paths based on case_log_path
        case_folder = os.path.dirname(case_log_path)
        error_log_path = os.path.join(case_folder, "error_log.txt")
        system_log_path = case_log_path
        matched_log_path = os.path.join(case_folder, "matched_log.txt")

        # Match errors with system log
        if os.path.exists(error_log_path) and os.path.exists(system_log_path):
            with open(error_log_path, "r") as f:
                error_lines = [line.strip() for line in f if "at " in line and len(line.strip().split("at ")) > 1]

            system_data = []
            with open(system_log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entry_time = datetime.strptime(entry["time"], "%m-%d-%Y %H:%M:%S.%f")
                        system_data.append((entry_time, entry))
                    except:
                        continue

            matched_entries = []
            dropout_times = []

            for error_line in error_lines:
                try:
                    ts_str = error_line.split("at ")[1]
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                    dropout_times.append(ts)


                    closest_entry = None
                    min_diff = timedelta(seconds=1)
                    for entry_time, entry in system_data:
                        diff = abs(entry_time - ts)
                        if diff <= min_diff:
                            min_diff = diff
                            closest_entry = entry

                    if closest_entry:
                        matched_entries.append(f"{error_line}\nSystem Info: {closest_entry}\n\n")
                    else:
                        matched_entries.append(f"{error_line}\nSystem Info: No close system log found\n\n")

                except Exception as e:
                    matched_entries.append(f"{error_line}\nSystem Info: Failed to parse timestamp ({e})\n\n")

            with open(matched_log_path, "w") as f:
                f.writelines(matched_entries)

            print(f"Matched log written to {matched_log_path}")
        else:
            print("Error or system log not found for matching.")

        
        # Step 1: Read the data from the file
        with open(local_path, 'r') as file:
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

        for ax in axs:
            for ts in dropout_times:
                start = ts - highlight_window
                end = ts + highlight_window
                ax.axvspan(start, end, color='pink', alpha=0.05)


        # Layout
        plt.tight_layout()
        # Save figure to the case folder
        case_folder = os.path.dirname(case_log_path)
        fig_path = os.path.join(case_folder, "system_metrics.png")
        plt.savefig(fig_path)
        print(f"System metrics plot saved to: {fig_path}")

        #plt.show()


    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

# Block until terminated
print("autossh.py is now blocking until terminated...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    cleanup_and_exit()
