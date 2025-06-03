# Arthrex Video Player & Real-Time Error Detection GUI

A Python GUI tool for video playback with real-time and file-based error detection using OpenCV and custom scripts. Supports SSH-based remote logging and system log analysis.

---

## Features

- Video file playback and real-time camera streaming
- Pluggable error detection scripts (real-time or offline)
- Adjustable preview mask for focusing on regions of interest
- Remote SSH script execution and system monitoring via `jtop` and `Fabric`
- View system logs from Jetson remotely in a GUI popup
- Detected anomalies are saved in a system logs folder

## Software Setup
1. Python 3.10+
2. Anaconda or other Python Environment Manager(optional)
3. Laptop running a Python IDE (VSCode highly preferred)

## Hardware Setup
1. Both the laptop and the Arthrex Synergy Vision console needs to be connected to the same wifi network.
2. Note the IP Address in the Synergy Vision Console Splash Screen. Another way to obtain the IP address is to open the Internet settings menu on the tablet.
3. Username and Password for Console is console-specific.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/.../ece188a-arthrex.git
   cd ece188a-arthrex
   ```

2. **Create a virtual environment (optional but recommended)**:

  For Windows users: Run installer.bat to automatically load all dependencies and create a venv if you don't have one already

   Using Python venv.
   ```bash
   python -m venv venv
   venv\Scripts\activate # On IOS: source venv/bin/activate  
   ```

   Using Anaconda Environment for Easier Dependency Management
   ```bash
   conda create --name ArthrexScripts python=3.10
   ```

4. **Install required packages**:

   Using Python venv
   ```bash
   pip install -r requirements.txt
   ```

   Using Anaconda Environment
   ```bash
   conda install conda-forge::opencv
   conda install anaconda::numpy
   conda install conda-forge::matplotlib
   conda install conda-forge::pillow
   conda install anaconda::scikit-image
   ...
   ```

---

## Python Required Libraries

* opencv-python — for video handling

* numpy — numerical computation

* matplotlib — plotting

* Pillow — image conversion for Tkinter

* scikit-image — image processing

* scipy — signal processing

* numba — performance optimization

* fabric — remote command execution

* keyboard — keypress detection

* pandas - dataset operations for logging

---

## Usage

Open PowerShell Window inside GUI Folder. Run:
```bash
python Final_GUI/final_gui.py
```

Or open in preferred Python IDE.

### Buttons Overview

* **Choose File**: Load a video file for playback.
* **Real-Time**: Activate live camera feed with error detection.
* **Play**: Start playback.
* **Reset**: Stop and clear current playback.
* **Open System Logs**: View logs downloaded from the remote system.
* **Run SSH Task**: Runs a remote monitoring script on the Jetson device via SSH and downloads system logs.

---

## Remote Execution Details

The `autossh.py` script runs a remote process using `Fabric`. Autossh python code is integrated inside the GUI. The user needs to press the "Run AutoSSH" allowing the user to use Python to send Linux Commands.

The script will prompt for:

* IP Address
* Username
* Password

The IP Address is found on the Arthrex Synergy Vision Console.

Either run `python3 jtopTestScript.py` on the remote Jetson or use the GUI.

## Contributors

* Zion Chang — [`@zionucsb`](https://github.com/zionucsb)
* Tony Korol - [`@TonyKorol1`](https://github.com/TonyKorol1)
* Will Peng — [`@willpeng945`](https://github.com/willpeng945)
* Sidney Lu — [`@sheteneu`](https://github.com/sheteneu)
* Jing Bian — [`@JingBian87`](https://github.com/JingBian87)
