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

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Requirements

* Python 3.7+

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

Run the main GUI script:

```bash
python gui_realtime/gui_maybe.py
```

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

---

## License

MIT License. See `LICENSE` file for details.

???

---

## Contributors

* Zion Chang — [`@zionucsb`](https://github.com/zionucsb)
* Tony Korol - [`@TonyKorol1`](https://github.com/TonyKorol1)

???
