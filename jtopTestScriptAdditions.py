import json
import time
import pytz
from datetime import timezone

import jtop

def datetime_to_ns(dt):
    pst = pytz.timezone('US/Pacific')
    # Ensure the datetime is in PST
    dt = dt.astimezone(pst)
    # Convert datetime to timestamp in seconds
    # timestamp = dt.strftime('%m-%d-%Y %H:%M:%S')
    timestamp = dt.strftime("%m-%d-%Y %H:%M:%S.%f")[:-3]
    return timestamp

def collect_and_send_metrics_to_file():
    file_path = "/home/arthrex/system_logs.txt"
    interval = 0.001
    with jtop.jtop() as jetson:
        # Start reading stats
        start_time = time.time()
        while True:
            data = jetson.stats
            performance_dict = dict()
            performance_dict["time"] = datetime_to_ns(data["time"])
            performance_dict["uptime"] = str(round(data["uptime"].total_seconds()/60, 2)) + " hours"
            performance_dict["cpu1"] = str(data["CPU1"]) + "%"
            performance_dict["cpu2"] = str(data["CPU2"]) + "%"
            performance_dict["cpu3"] = str(data["CPU3"]) + "%"
            performance_dict["cpu4"] = str(data["CPU4"]) + "%"
            performance_dict["cpu6"] = str(data["CPU6"]) + "%"
            performance_dict["cpu7"] = str(data["CPU7"]) + "%"
            performance_dict["cpu8"] = str(data["CPU8"]) + "%"
            performance_dict["gpu"] = str(data["GPU"]) + "%"
            performance_dict["ram"] = str(data["RAM"])[2:4] + "%"
            performance_dict["swap"] = data["SWAP"]

            #User Defined Additions
            performance_dict["Temp Thermal"] = str(data["Temp thermal"]) + "C"
            performance_dict["Temp CPU"] = str(data["Temp CPU"]) + "C"
            performance_dict["Temp GPU"] = str(data["Temp GPU"]) + "C"
            performance_dict["Fan Speed"] = str(data["Fan pwmfan0"]) + "RPM"
            performance_dict["CPU Power"] = str(data["Power CPU"] * 0.001) + "W"
            performance_dict["GPU Power"] = str(data["Power GPU"] * 0.001) + "W"

            # Serialize the dictionary to a string (e.g., JSON)
            performance_data_str = json.dumps(performance_dict)

            # Write output to file
            with open(file_path, "a") as file:
                file.write(performance_data_str + "\n")
                file.close()

            # Wait for the next interval
            time.sleep(interval)

if __name__ == "__main__":
    collect_and_send_metrics_to_file()

