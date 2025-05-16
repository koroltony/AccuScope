from fabric import Connection, Config
import getpass
import time
import sys
import signal
import shutil
from datetime import datetime, timedelta
import json
import os

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
conn.run('nohup python3 jtopTestScript.py > system_logs.txt 2>&1 &', pty=False)
print("Remote script running in background.")

# Define shutdown logic
def cleanup_and_exit(signum=None, frame=None):
    print("Shutting down remote monitoring...")
    try:
        conn.run("pkill -f jtopTestScript.py")
        print("Remote script killed.")

        remote_path = './system_logs.txt'
        local_path = './arthrex_system_logs.txt'
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
                dropout_errors = [line.strip() for line in f if line.startswith("Dropout Error at")]

            system_data = []
            with open(system_log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        entry_time = datetime.strptime(entry["time"], "%m-%d-%Y %H:%M:%S")
                        system_data.append((entry_time, entry))
                    except:
                        continue

            matched_entries = []
            for error_line in dropout_errors:
                try:
                    ts_str = error_line.replace("Dropout Error at ", "")
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")

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
