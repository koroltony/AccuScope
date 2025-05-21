# Arthrex Video Player & Real-Time Error Detection GUI

A Python GUI tool for video playback with real-time and file-based error detection using OpenCV and custom scripts. Supports SSH-based remote logging and system log analysis.

---

## Features

- Video file playback and real-time camera streaming
- Pluggable error detection scripts (real-time or offline)
- Adjustable preview mask for focusing on regions of interest
- Remote SSH script execution and system monitoring via `jtop`
- View system logs from Jetson remotely in a GUI popup

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/.../ece188a-arthrex.git
   cd ece188a-arthrex
   ```

2. **Create a virtual environment (optional but recommended)**:

   Using Python venv.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   Using Anaconda Environment for Easier Dependency Management
   ```bash
   conda create --name ArthrexScripts python=3.10
   ```

3. **Install required packages**:

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

## Requirements

* Python 3.10+

* opencv-python — for video handling

* numpy — numerical computation

* matplotlib — plotting

* Pillow — image conversion for Tkinter

* scikit-image — image processing

* scipy — signal processing

* numba — performance optimization

* fabric — remote command execution

* keyboard — keypress detection

---

## Usage

Open PowerShell Window inside GUI Folder. Run:
```bash
python gui_realtime/gui.py
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

The `autossh.py` script runs a remote process using `Fabric`, capturing and downloading system metrics.

The script will prompt for:

* IP Address
* Username
* Password

And run `jtopTestScript.py` on the remote Jetson.

## Contributors

* Zion Chang — [`@zionucsb`](https://github.com/zionucsb)
* Tony Korol - [`@TonyKorol1`](https://github.com/TonyKorol1)
* Will Peng — [`@willpeng945`](https://github.com/willpeng945)
* Sidney Lu — [`@sheteneu`](https://github.com/sheteneu)
* Jing Bian — [`@JingBian87`](https://github.com/JingBian87)
