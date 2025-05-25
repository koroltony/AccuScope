import sys
import os
import tkinter as tk
import keyboard
import subprocess
import signal
import json
from datetime import datetime
import threading
import cv2
import numpy as np
import time
# import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tkinter import simpledialog, filedialog, ttk
# from tkinter import Tk, Canvas, Text, Button, Label, Entry, StringVar, END
# from skimage import measure
# from skimage.draw import disk
import importlib

# Function to load the correct error detection scripts from a fixed directory
def load_error_detection_scripts():
    base_folder = "Consolidated/source_update"
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", base_folder))

    if folder_path not in sys.path:
        sys.path.append(folder_path)

    modules = {
        "green": "greenVectorizedSolution",
        "magenta": "magentaScreen",
        "black": "dropoutScreen",
        "highlights": "highlights",
        "frozen": "lagff15",
        "mask": "auto_mask",
        "pano": "panoto70fcn",
        "general": "general_detection"
    }

    loaded_scripts = {}
    for key, module_name in modules.items():
        try:
            loaded_scripts[key] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            print(f"Warning: {module_name}.py not found in {folder_path}")

    return loaded_scripts

def cv2_to_tk(img, scale=0.6):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return ImageTk.PhotoImage(Image.fromarray(img))

currentFrame = 1
window_size = 10
frozen_frame_buffer = []


# RealTimeMasking provides an interactive real-time mask editing interface 
# for live video streams. It allows users to:
# - Preview and adjust a binary mask on live frames
# - Use keyboard controls to shrink ('s') or grow ('g') the mask
# - Toggle raw video mode ('r') and confirm the final mask ('k')
# - Overlay mask edges in red for visual feedback
# - Return the final adjusted mask after confirmation

class RealTimeMasking:
    def __init__(self, video_source, scripts):
        self.cap = video_source
        self.scripts = scripts
        self.kernel = np.ones((3, 3), np.uint8)
        self.frame_counter = 0
        self.update_interval = 1
        self.keypress = False
        self.keypresg = False
        self.shrunk_mode = False
        self.running = True # Flag to control the loop
        self.raw_mode = False  # If True, masking and overlays are skipped


        # Inside RealTimeMasking.__init__
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            raise ValueError("Could not read a valid frame from the camera. Is the virtual camera running?")
            
        _, self.curr_frame = self.cap.read()
        self.lmask = self.scripts["mask"].create_mask(self.curr_frame)
        self.lmask = self.lmask.astype(np.uint8)
        self.shrunk_mask = self.lmask.copy()
        self.edges_red = np.zeros_like(self.curr_frame)

    def is_running(self):
        return self.running

    def show_controls_window(self):
        help_img = np.ones((200, 400, 3), dtype=np.uint8) * 255  # White background
        font = cv2.FONT_HERSHEY_SIMPLEX

        instructions = [
            "Controls:",
            "'s' - Shrink Mask",
            "'g' - Grow Mask",
            "'r' - Toggle Raw Video",
            "'k' - Confirm/Exit",
        ]

        for i, line in enumerate(instructions):
            cv2.putText(help_img, line, (10, 30 + i * 30), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("Controls", help_img)


    def update(self, frame, footage_label=None):
        if not self.running:
            return None
    
        self.curr_frame = frame
        self.frame_counter += 1

        self.show_controls_window()

        # Raw video mode toggle
        if keyboard.is_pressed('r'):
            self.raw_mode = not self.raw_mode
            print("Raw mode:", "ON" if self.raw_mode else "OFF")
            time.sleep(0.3)

        if self.raw_mode:
            
            if footage_label is not None:

                combined = np.hstack((self.curr_frame, self.curr_frame))  # Side-by-side
            
                # Convert to PhotoImage
                tk_combined = cv2_to_tk(combined)
            
                # Show in single label
                footage_label.configure(image=tk_combined)
                footage_label.image = tk_combined
                
            if cv2.waitKey(1) & 0xFF == ord('k') or keyboard.is_pressed('k'):
                print("Raw mode active. Exiting mask preview.")
                self.running = False
            # Return dummy zero mask to avoid Numba crash
            dummy_mask = np.zeros(self.curr_frame.shape[:2], dtype=np.uint8)
            return dummy_mask

        # Update mask every 10 frames if 'k' not pressed
        if not keyboard.is_pressed('k'):
            if self.frame_counter % self.update_interval == 0 or self.frame_counter == 1:
                self.lmask = self.scripts["mask"].create_mask(self.curr_frame)
                self.lmask = self.lmask.astype(np.uint8)
            if not self.shrunk_mode:
                self.shrunk_mask = self.lmask.copy()

        if keyboard.is_pressed('s') and not self.keypress:
            self.keypress = True
        if self.keypress and not keyboard.is_pressed('s'):
            self.shrunk_mask = cv2.erode(self.shrunk_mask, self.kernel, iterations=3)
            self.shrunk_mode = True
            self.keypress = False

        if keyboard.is_pressed('g') and not self.keypresg:
            self.keypresg = True
        if self.keypresg and not keyboard.is_pressed('g'):
            self.shrunk_mask = cv2.dilate(self.shrunk_mask, self.kernel, iterations=3)
            self.shrunk_mode = True
            self.keypresg = False

        # Update red edge overlay
        if self.frame_counter % self.update_interval == 0:
            edges = cv2.Canny(self.curr_frame, 100, 200)
            masked_edges = cv2.bitwise_and(edges, edges, mask=self.shrunk_mask)
            self.edges_red = cv2.merge([
                np.zeros_like(edges),
                np.zeros_like(edges),
                masked_edges
            ])

        pano_green = cv2.merge([
            np.zeros_like(self.shrunk_mask),
            self.shrunk_mask,
            np.zeros_like(self.shrunk_mask)
        ])
        pano_green = (pano_green * 0.3).astype(np.uint8)
        pano_overlay = cv2.addWeighted(pano_green, 0.5, self.edges_red, 1.0, 0)
        
        if footage_label is not None:
            # Convert both masks to RGB
            mask_rgb = cv2.merge([self.lmask] * 3)
            combined = np.hstack((mask_rgb, pano_overlay))  # Side-by-side
        
            # Convert to PhotoImage
            tk_combined = cv2_to_tk(combined)
        
            # Show in single label
            footage_label.configure(image=tk_combined)
            footage_label.image = tk_combined

        # Exit when 'k' is clicked
        if cv2.waitKey(1) & 0xFF == ord('k') or keyboard.is_pressed('k'):
            print("Both masks set.")
            self.running = False  # Set the flag to False to exit the loop
            
            # Remove GUI preview images
            if footage_label:
                footage_label.destroy()
            cv2.destroyAllWindows()
            
            return self.shrunk_mask
        
        return self.shrunk_mask

# App is the main Tkinter application window that manages two primary views:
# - StartScreen: the initial UI where the user selects or configures options
# - VideoPlayer: the main interface for video playback and analysis
# It initializes both views, places them in the same layout grid, and uses 
# `tkraise()` to switch between them.

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Arthroscope Analysis Tool")
        self.geometry("1000x700")  # Adjust as needed

        # Make grid cells expand with window
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Initialize both screens
        self.start_screen = StartScreen(self)
        self.video_player = VideoPlayer(self)

        # Place both screens in the same grid
        self.start_screen.grid(row=0, column=0, sticky="nsew")
        self.video_player.grid(row=0, column=0, sticky="nsew")

        # Show start screen first
        self.show_start()

    def show_start(self):
        self.start_screen.tkraise()

    def show_main(self):
        self.video_player.tkraise()

# StartScreen is the initial frame shown to the user when the app launches.
# It displays a welcome message, logo, and buttons to either start the analysis 
# or open a help window. The layout is centered and styled for a clean UI.

class StartScreen(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.configure(bg="#1e1e1e")

        # Configure grid for full window expansion
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Internal container to center main content
        container = tk.Frame(self, bg="#1e1e1e")
        container.grid(row=0, column=0, sticky="nsew")

        # Load and place the logo at the top
        try:
            logo_image = Image.open("newlogoupsacled 2.png")  # Adjust path if needed
            logo_image = logo_image.resize((490, 200))
            self.logo_photo = ImageTk.PhotoImage(logo_image)

            logo_label = tk.Label(container, image=self.logo_photo, bg="#1e1e1e")
            logo_label.pack(pady=(30, 10))
        except Exception as e:
            print(f"Error loading logo: {e}")

        # Welcome label under the logo
        label = tk.Label(container, text="Welcome to the Arthroscope Video Analyzer",
                        font=("Arial", 20), bg="#1e1e1e", fg="white")
        label.pack(pady=10)

        # Start Analysis button
        start_button = tk.Button(container, text="Start Analysis",
                                font=("Arial", 16), command=master.show_main)
        start_button.pack(pady=20)

        # Help button in top-right corner of main screen
        self.help_button = tk.Button(self, text="Help", command=self.open_help_window)
        self.help_button.grid(row=0, column=1, padx=10, pady=10, sticky="ne")


    def open_help_window(self):
        help_win = tk.Toplevel(self)
        help_win.title("Help / User Guide")
        help_win.geometry("500x300")

        help_text = tk.Text(help_win, wrap=tk.WORD, font=("Arial", 11))
        help_text.insert(tk.END, """
    Welcome to the Arthroscope Video Analyzer!

    Quick Start:
    1. Press "Start Analysis" to head to the Main Video Player screen.
    2(a). Click "Choose File" to begin post processed video detection.
    2(b). Click "Real-Time" to begin live footage detection.
    3(a). Click "Play" to analyze the footage. 
    3(b). Adjust the preview mask as needed if "Real-Time".
    4. View detected errors in the log panel on the right.

    Extras:
    - "Toggle Camera" to switch inputs (0, 1, 2).
    - "Run AutoSSH" to connect to Jetson remotely.
    - "Open System Logs" to view system performance data.

    Each session creates a 'Case' folder with logs and saved video.

    For full documentation, see: README.md
    """)
        help_text.config(state=tk.DISABLED)
        help_text.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)


# VideoPlayer is a Tkinter-based GUI class for loading, playing, and analyzing 
# both real-time and file-based videos. It supports error detection using custom 
# scripts, real-time system logging via SSH, dynamic masking, and automatic case 
# folder creation for saving logs and video outputs. The class also provides 
# functions for resetting the UI, toggling cameras, managing logs, and handling 
# resource cleanup.

class VideoPlayer(tk.Frame):
    def __init__(self, master):
        
        super().__init__(master)
        #self.configure(bg="white")
        self.source = 0


        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        label = tk.Label(self, text="Video Player Placeholder", font=("Arial", 16))
        label.grid(row=1, column=0, columnspan=3, pady=10)

        self.lock = threading.Lock()
        self.is_realtime = False
        self.scripts = load_error_detection_scripts()

        # Video playback canvas
        self.canvas = tk.Canvas(self, bg="black", width=640, height=480)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        # Error log
        self.error_log = tk.Text(self, height=20)
        self.error_log.grid(row=0, column=3, sticky='ne', padx=5, pady=5)
        self.grid_columnconfigure(3, weight=2)

        # Progress bar and label
        self.progress_label = tk.Label(self, text="0:00 / 0:00")
        self.progress_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=640, mode='determinate')
        self.progress_bar.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        # Buttons
        self.choose_button = tk.Button(self, text="Choose File", command=self.choose_file)
        self.choose_button.grid(row=2, column=0, padx=5, pady=5)

        self.realtime_button = tk.Button(self, text="Real-Time", command=self.realtime_video)
        self.realtime_button.grid(row=3, column=0, padx=5, pady=5)

        self.play_button = tk.Button(self, text="Play", command=self.play_video, state=tk.DISABLED)
        self.play_button.grid(row=2, column=1, padx=5, pady=5)

        self.reset_button = tk.Button(self, text="Reset", command=self.reset_gui)
        self.reset_button.grid(row=3, column=1, padx=5, pady=5)

        self.autossh_button = tk.Button(self, text="Run AutoSSH", command=self.run_autossh)
        self.autossh_button.grid(row=3, column=2, padx=5, pady=5)

        self.open_logs_button = tk.Button(self, text="Open System Logs", command=self.open_system_logs)
        self.open_logs_button.grid(row=2, column=2, padx=5, pady=5)

        self.toggle_camera_button = tk.Button(self, text="Toggle Camera Source", command=self.toggle_camera)
        self.toggle_camera_button.grid(row=4, column=0, padx=5, pady=5)
        
        # Sensitivity control
        self.sensitivity = tk.DoubleVar(value=0.2)
        self.sensitivity_slider = tk.Scale(
            self,
            from_=0.0,
            to=0.5,
            resolution=0.05,
            orient="horizontal",
            label="Error Discovery Sensitivity",
            variable=self.sensitivity,
            length = 200
        )
        self.sensitivity_slider.grid(row=5, column=0, columnspan=3, padx=5, pady=10)
        
        self.end_real_time_button = tk.Button(self, text="Finish Real Time", command=self.stop_realtime)
        self.end_real_time_button.grid(row=4, column=1, padx=5, pady=5)

        # Status label
        self.status_label = tk.Label(self, text="No video detected.\nRemaining time: N/A", fg="red")
        self.status_label.grid(row=1, column=3, padx=5, pady=5)

        self.footage_mask_label = tk.Label(self)
        self.footage_mask_label.grid(row=2, column=3, padx=0, pady=0)

        # Internal state
        self.cap = None
        self.image_on_canvas = None
        self.play_flag = False
        self.file_path = None
        self.frozen_frame_flags = []
        self.start_time = None
        self.fps = 0

        self.case_number, self.timestamp = self.create_new_case_folder()  # Initialize the new case number and timestamp
        self.case_folder = os.path.join("Cases", f"Case_{self.case_number}_{self.timestamp}")
        os.makedirs(self.case_folder, exist_ok=True)  # Create the case folder
        # Initialize log files
        self.error_log_path = os.path.join(self.case_folder, "error_log.txt")
        self.system_log_path = os.path.join(self.case_folder, "system_log.txt")
        self.matched_log_path = os.path.join(self.case_folder, "matched_log.txt")
        self.raw_video_path = os.path.join(self.case_folder, 'raw_video.mp4')
        self.error_video_path = os.path.join(self.case_folder, 'error_video.mp4')
        self.create_log_files()
        self.saved_vid = cv2.VideoWriter(self.raw_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (640,480))
        self.error_vid = cv2.VideoWriter(self.error_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (2*640,480))


        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.error_frame = self.black_frame.copy()
        self.error_counter = 0
        self.error_text = "No Error"

    def create_new_case_folder(self):
            # Check if the "cases" folder exists, create if not
            if not os.path.exists("Cases"):
                os.makedirs("Cases")

            # Get the highest case number to determine the next available case number
            case_folders = [d for d in os.listdir("Cases") if os.path.isdir(os.path.join("Cases", d))]
            case_numbers = [int(folder.split('_')[1]) for folder in case_folders if folder.startswith("Case_")]

            # If there are no case folders yet, start with case 1
            if case_numbers:
                new_case_number = max(case_numbers) + 1
            else:
                new_case_number = 1
            
            # Get the current timestamp in the format "YYYYMMDD_HHMMSS"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            return new_case_number, timestamp

    def create_log_files(self):
            # Create or reset the error and system log files inside the case folder
            with open(self.error_log_path, 'w') as f:
                f.write(f"Error Log for case {self.case_number}\n")
            with open(self.system_log_path, 'w') as f:
                f.write(f"System Log for case {self.case_number}\n")
            with open(self.matched_log_path, 'w') as f:
                f.write(f"Matched Log for case {self.case_number}\n")

    def run_autossh(self):
        try:
            """ip = simpledialog.askstring("SSH IP", "Enter the remote IP address:", parent = self.root)
            username = simpledialog.askstring("SSH Username", "Enter the username:", parent = self.root)
            password = simpledialog.askstring("SSH Password", "Enter the password:", show="*", parent = self.root)"""

            ip = simpledialog.askstring("SSH IP", "Enter the remote IP address:", parent = self)
            username = simpledialog.askstring("SSH Username", "Enter the username:", parent = self)
            password = simpledialog.askstring("SSH Password", "Enter the password:", show="*", parent = self)

            if not all([ip, username, password]):
                self.status_label.config(text="SSH details incomplete.", fg="red")
                return

            # Pass the data to autossh.py
            autossh_path = os.path.join(os.path.dirname(__file__), "autossh.py")
            self.autossh_process = subprocess.Popen(
                [sys.executable, "-W", "ignore", autossh_path, ip, username, password, self.system_log_path],
                shell=False,
                start_new_session=True
            )
            self.status_label.config(text="autossh.py launched with user input.", fg="green")

        except Exception as e:
            self.status_label.config(text=f"Failed to run autossh.py: {e}", fg="red")

    def open_system_logs(self):
        try:
            system_log_path = os.path.join(os.path.dirname(__file__), "arthrex_system_logs.txt")
            with open(system_log_path, "r") as file:
                content = file.read()

            # Create a new popup window
            """log_window = tk.Toplevel(self.root)"""
            log_window = tk.Toplevel(self)

            log_window.title("System Logs")
            log_window.geometry("600x400")

            # Frame to contain Text widget and Scrollbar
            frame = tk.Frame(log_window)
            frame.pack(expand=True, fill=tk.BOTH)

            text_widget = tk.Text(frame, wrap=tk.WORD)
            text_widget.insert(tk.END, content)
            text_widget.config(state=tk.DISABLED)
            text_widget.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

            scrollbar = tk.Scrollbar(frame, command=text_widget.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.config(yscrollcommand=scrollbar.set)

        except FileNotFoundError:
            self.status_label.config(text="system_logs.txt not found.", fg="red")
        except Exception as e:
            self.status_label.config(text=f"Error opening logs: {e}", fg="red")
            
        
    def toggle_camera(self):
        sources = [0, 1, 2]
        
        if self.saved_vid:    
            self.saved_vid.release()
            os.remove(self.raw_video_path)
            self.saved_vid = cv2.VideoWriter(self.raw_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (640,480))
            
        if self.error_vid:    
            self.error_vid.release()
            os.remove(self.error_video_path)
            self.error_vid = cv2.VideoWriter(self.error_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (2*640,480))


        # Repeatedly try sources until one works:
            
        self.cap = cv2.VideoCapture(self.source)
        
        self.source = (self.source + 1) % len(sources)
        self.cap = cv2.VideoCapture(self.source)
    

        # Check the FPS to see if it is a valid source (0 is not valid)
        
        while True:
            
            if self.cap.get(cv2.CAP_PROP_FPS) == 0:
                print(self.source)
                self.source = (self.source + 1) % len(sources)
                self.cap = cv2.VideoCapture(self.source)
            else:
                break

    def on_close(self):
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                print("Camera released")

            # Release video writers if they exist
            try:
                self.saved_vid.release()
                print("Saved video released")
            except Exception as e:
                print(f"Error releasing saved_vid: {e}")

            try:
                self.error_vid.release()
                print("Error video released")
            except Exception as e:
                print(f"Error releasing error_vid: {e}")

        except Exception as e:
            print(f"Exception in cleanup: {e}")

        # Terminate autossh.py process if running
        if hasattr(self, 'autossh_process') and self.autossh_process.poll() is None:
            try:
                print("Terminating autossh process...")
                self.autossh_process.send_signal(signal.CTRL_C_EVENT)
                print("Sent CTRL_C_EVENT, waiting up to 5s for process to exit...")
                
                # With this:
                for _ in range(10):
                    if self.autossh_process.poll() is not None:
                        print("autossh process terminated.")
                        break
                    time.sleep(0.5)
                else:
                    print("autossh process did not terminate. Killing...")
                    self.autossh_process.kill()
                    self.autossh_process.wait()
                    print("autossh process killed.")

                print("autossh process terminated.")
            except subprocess.TimeoutExpired:
                print("autossh process did not terminate in time. Killing it...")
                self.autossh_process.kill()
                self.autossh_process.wait()
                print("autossh process killed.")
            except Exception as e:
                print(f"Error terminating autossh process: {e}")

        # Close all OpenCV and GUI windows
        try:
            cv2.destroyAllWindows()
            print("OpenCV windows closed")
        except Exception as e:
            print(f"Error closing OpenCV windows: {e}")

        try:
            self.master.destroy()
            print("Tkinter window closed")
        except Exception as e:
            print(f"Error destroying Tkinter window: {e}")


    def update_status(self, message, remaining_time = "N/A", color="black"):
        # self.status_label.config(text=f"{message}\nRemaining time: {remaining_time}", fg=color)
        """Updated to handle real-time mode."""
        if self.is_realtime:
            self.status_label.config(text=f"{message}", fg = "blue")  # No remaining time in real-time
        else:
            self.status_label.config(text=f"{message}\nRemaining time: {remaining_time}", fg=color)


    def update_log(self, message, color="black"):
        self.error_log.insert(tk.END, message + '\n')
        self.error_log.see(tk.END)  # Scroll to the end

    def save_log_to_file(self, filename="error_log.txt"):
        with open(filename, 'w') as file:
            file.write(self.error_log.get("1.0", tk.END))

    def reset_gui(self):
        # Stop any playback or video capture
        self.play_flag = False
        if self.cap:
            self.cap.release()
            self.cap = None
            
            
        if self.saved_vid:    
            self.saved_vid.release()
            os.remove(self.raw_video_path)
            self.saved_vid = cv2.VideoWriter(self.raw_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (640,480))
            
        if self.error_vid:    
            self.error_vid.release()
            os.remove(self.error_video_path)
            self.error_vid = cv2.VideoWriter(self.error_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (2*640,480))

        # Clear canvas
        self.canvas.delete("all")
        self.image_on_canvas = None

        # Reset error log
        self.error_log.delete(1.0, tk.END)

        # Reset progress
        self.progress_bar["value"] = 0
        self.progress_label.config(text="0:00 / 0:00")

        # Reset status
        self.status_label.config(text="No video detected.\nRemaining time: N/A", fg="red")

        # Reset file and flags
        self.file_path = None
        self.frozen_frame_flags = []
        self.start_time = None
        self.fps = 0

        # Re-enable buttons if needed
        self.play_button.config(state=tk.DISABLED)

        self.is_realtime = False
        self.shrunk_mask = None
        self.dummy_mask = None

        self.footage_mask_label.destroy()  # Optional: if you know it's still around
        self.footage_mask_label = tk.Label(self)

        self.footage_mask_label.grid(row=2, column=3, padx=0, pady=0)
        
    def stop_realtime(self):

        # Stop any playback or video capture
        self.play_flag = False
        if self.cap:
            self.cap.release()
            self.cap = None
            self.saved_vid.release()
            self.error_vid.release()
            if not self.is_realtime:
                os.remove(self.raw_video_path)
            
        self.update_status("Finished playing video.", "0:00", "blue")
        self.canvas.delete("all")

        # After processing is done
        self.update_log("Video is done being processed...")
        self.save_log_to_file(self.error_log_path)

        # Clear canvas
        self.canvas.delete("all")
        self.image_on_canvas = None

        # Reset progress
        self.progress_bar["value"] = 0
        self.progress_label.config(text="0:00 / 0:00")

        # Reset status
        self.status_label.config(text="No video detected.\nRemaining time: N/A", fg="red")

        # Reset file and flags
        self.file_path = None
        self.frozen_frame_flags = []
        self.start_time = None
        self.fps = 0

        # Re-enable buttons if needed
        self.play_button.config(state=tk.DISABLED)

        self.is_realtime = False
        self.shrunk_mask = None
        self.dummy_mask = None

        self.footage_mask_label.destroy()  # Optional: if you know it's still around
        """self.footage_mask_label = tk.Label(self.root)"""
        self.footage_mask_label = tk.Label(self)

        self.footage_mask_label.grid(row=2, column=3, padx=0, pady=0)

    def choose_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            if os.path.splitext(file_path)[1].lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                # Release previous video capture if one exists
                if self.cap is not None and self.cap.isOpened():
                    self.cap.release()
                    self.cap = None  # Reset capture

                # Reset the canvas display
                self.canvas.delete("all")  
                self.image_on_canvas = None  

                # Store the new file path and enable play button
                self.file_path = file_path
                self.update_status("Video is currently loaded.", "N/A", "green")
                self.play_button.config(state=tk.NORMAL)
            else:
                self.update_status("Improper file format.", "N/A", "red")
                self.play_button.config(state=tk.DISABLED)
        else:
            self.update_status("No file chosen.", "N/A", "red")

    def play_video(self):
        if self.file_path and not self.is_realtime:
            self.is_realtime = False  # Set mode to file processing
            self.scripts = load_error_detection_scripts()

            # Release previous capture if it exists
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.cap = None  # Reset the cap to avoid stale reference

            self.cap = cv2.VideoCapture(self.file_path)

            _, initial_frame = self.cap.read()
            initial_frame = cv2.resize(initial_frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            lmask = self.scripts["mask"].create_mask(initial_frame).astype(np.uint8)

            # Set the current_mask attribute
            self.current_mask = lmask

            if not self.cap.isOpened():
                self.update_status("Error with file.", "N/A", "red")
                return

            self.start_time = time.time()  # Post-processing does not need real-time tracking
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Precompile numba code for fast execution:
                
            _ = self.scripts["black"].checkBlackFrame_numba(initial_frame,lmask)
            _ = self.scripts["green"].checkGreenFrame_numba(initial_frame)
            _ = self.scripts["magenta"].checkMagentaFrame_numba(initial_frame)

            # file_name = os.path.basename(self.file_path)  # Extract just the file name
            self.update_status("Video is currently playing.", "Calculating...", "green")
            self.update_log(f"Video file '{self.file_path}' is now being processed...", "blue")  # Add log update with the whole path
            #self.update_log(f"Video file '{file_name}' is now being processed...", "blue")  # Add log update with only the file name

            self.play_flag = True
            self.process_video_frame()
        else:
            self.update_status("No video detected.", "N/A", "red")



    def realtime_video(self):
        # self.root.update()
        #print("DEBUG: switching to real time mode")
        self.file_path = None
        self.play_flag = True
        self.is_realtime = True


        # Clear the canvas
        self.canvas.delete("all")
        self.image_on_canvas = None  # Reset canvas display
        #print("DEBUG: canvas cleared")
        
        # Release previous capture if it exists
        if self.cap is not None:
            #print("DEBUG: releasing previous video capture")
            self.cap.release()
            self.cap = None  # Reset to avoid stale reference
        
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap or not self.cap.isOpened():
            self.cap = None
            self.update_status("Error with webcam.", "N/A", "red")
            return
        
        self.scripts = load_error_detection_scripts()
        self.masking = RealTimeMasking(self.cap, self.scripts)  # Instantiate once

        
        self.is_realtime = True  # Set mode to real-time

        # self.load_scripts(real_time=True)
        self.masking = RealTimeMasking(self.cap, self.scripts)  # Safe to instantiate now
        self.running = True

        self.play_flag = True  # Make sure this is set before looping
        self.start_time = time.time()
        self.fps = 60  # Approximate real-time FPS
        #print("DEBUG: webcam opened successfully")
        self.update_status("Real-time video is currently playing.", "green")
        
        self.process_video_frame()

    def process_video_frame(self):
        #print("DEBUG: processing video frame")
        if self.play_flag and self.cap is not None:
            ret, frame = self.cap.read()
            # print(f" frame retrieved: {ret}")
            
            if ret:
                if self.is_realtime:
                    self.saved_vid.write(frame)
                #print("DEBUG: frame read successfully")
                frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR)
                if hasattr(self, 'masking') and self.masking.is_running():
                    self.current_mask = self.masking.update(frame, self.footage_mask_label)
                self.detect_and_display_errors(frame)

                '''
                # Get actual video time (more accurate than frame count method)
                current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert ms to seconds
                total_time = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
                remaining_time = total_time - current_time

                # Update progress bar and labels
                self.progress_bar['value'] = (current_time / total_time) * 100
                self.progress_label.config(
                    text=f"{int(current_time//60)}:{int(current_time%60):02d} / {int(total_time//60)}:{int(total_time%60):02d}"
                )
                self.update_status("Video is currently playing.", f"{int(remaining_time//60)}:{int(remaining_time%60):02d}", "green")

                # Schedule the next frame processing without enforcing real-time playback
                self.root.after(16, self.process_video_frame)
                '''


                if not self.is_realtime:
                    # Update the progress bar and check if the video is playing in real time
                    current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    
                    total_time = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
                        
                    self.progress_bar['value'] = (current_time / total_time) * 100
                    self.progress_label.config(text=f"{int(current_time//60)}:{int(current_time%60):02d} / {int(total_time//60)}:{int(total_time%60):02d}")

                    # Check if video is playing in real time and calculate time discrepancy
                    elapsed_real_time = time.time() - self.start_time
                    speed_ratio = current_time / elapsed_real_time
                    percentage = speed_ratio * 100
                if self.is_realtime:
                    self.update_status("Video playing in real time.")
                else:
                    if abs(elapsed_real_time - current_time) < 1:
                        self.update_status(f"Video playing in real time. ({percentage:.2f}%)", "N/A", "green")
                    else:
                        remaining_real_time = (total_time - current_time) / speed_ratio
                        self.update_status(f"Video not playing in real time. ({percentage:.2f}%)", f"{int(remaining_real_time//60)}:{int(remaining_real_time%60):02d}", "red")

                """self.root.after(5, self.process_video_frame)"""
                self.after(5, self.process_video_frame)

                #self.root.after(16, self.process_video_frame)
            
            else:
                #print("DEBUG: no frame captured")
                self.saved_vid.release()
                self.error_vid.release()
                if not self.is_realtime:
                    os.remove(self.raw_video_path)
                self.cap.release()
                self.update_status("Finished playing video.", "0:00", "blue")
                self.canvas.delete("all")

                if not self.is_realtime:
                    # After processing is done
                    file_name = self.file_path # File path included
                    # file_name = os.path.basename(self.file_path)  # Extract just the file name
                    self.update_log(f"Video file '{file_name}' is done being processed...", "blue")  # Add log update
                    self.save_log_to_file(self.error_log_path)





    def detect_and_display_errors(self, frame):
        
        '''
        Masking the entire footage is unnecessary, we only mask for dropout
        
        if hasattr(self, 'current_mask') and self.current_mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=self.current_mask)
            mask_applied = True
        else:
            mask_applied = False
        '''
        # mask_applied = False

        with self.lock:
            error_duration = 2*self.fps
            currentFrame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # e.g., 2025-05-05 14:32:21.123

        # Use dynamically loaded modules
        if "green" in self.scripts and self.scripts["green"]:

            if self.scripts["green"].checkGreenFrame_numba(frame) == 1:
                if self.is_realtime:
                    self.error_text = f"Full Green Screen Error at {timestamp}"
                else:
                    self.error_text = f"Full Green Screen Error Detected at {round((currentFrame/self.fps), 4)} seconds\nand frame: {currentFrame}"
                self.update_log(self.error_text, "red")
                self.error_frame = frame.copy()
                self.error_counter = error_duration


            if self.scripts["green"].checkGreenFrame_numba(frame) == 2:
                if self.is_realtime:
                    self.error_text = f"Corner Green Screen Error at {timestamp}"
                else:
                    self.error_text = f"Corner Green Screen Error Detected at {round((currentFrame/self.fps), 4)} seconds\nand frame: {currentFrame}"
                self.update_log(self.error_text, "red")
                self.error_frame = frame.copy()
                self.error_counter = error_duration

        if "magenta" in self.scripts and self.scripts["magenta"]:
            if self.scripts["magenta"].checkMagentaFrame_numba(frame):
                if self.is_realtime:
                    self.error_text = f"Magenta Screen Error at {timestamp}"
                else:
                    self.error_text = f"Magenta Screen Error Detected at {round((currentFrame/self.fps), 4)} seconds\nand frame: {currentFrame}"
                self.update_log(self.error_text, "red")
                self.error_frame = frame.copy()
                self.error_counter = error_duration

        if "black" in self.scripts and self.scripts["black"]:
            if getattr(self,'masking',None) and self.masking.raw_mode:
                if self.scripts["black"].checkDropoutNoMask(frame):
                    if self.is_realtime:
                        self.error_text = f"Dropout Error at {timestamp}"
                    else:
                        self.error_text = f"Dropout Error Detected at {round((currentFrame/self.fps), 4)} seconds\nand frame: {currentFrame}"
                    self.update_log(self.error_text, "red")
                    self.error_frame = frame.copy()
                    self.error_counter = error_duration
            else:
                if self.scripts["black"].checkBlackFrame_numba(frame, self.current_mask):
                    if self.is_realtime:
                        self.error_text = f"Dropout Error at {timestamp}"
                    else:
                        self.error_text = f"Dropout Error Detected at {round((currentFrame/self.fps), 4)} seconds\nand frame: {currentFrame}"
                    self.update_log(self.error_text, "red")
                    self.error_frame = frame.copy()
                    self.error_counter = error_duration
                    

        #if "highlights" in self.scripts and self.scripts["highlights"]:
        #    if self.scripts["highlights"].checkHighlightsFrame(frame):
        #        error_text = f"Highlight Shimmer Detected at {round((currentFrame/self.fps), 4)} seconds"
        #        self.update_log(error_text, "red")
        '''
        if "frozen" in self.scripts and self.scripts["frozen"]:
            if hasattr(self, "prev_frame") and self.scripts["frozen"].detect_frozen_frame(self.prev_frame, frame):
                error_text = "Frozen Frame" if self.is_realtime else f"Frozen Frame Detected at {round((currentFrame/self.fps), 4)} seconds"
                self.update_log(error_text, "red")
        '''
        if "frozen" in self.scripts and self.scripts["frozen"]:
            if hasattr(self, "prev_frame") and self.scripts["frozen"].detect_frozen_frame(self.prev_frame, frame):
                #print(f"Frozen Frame at {time_stamp:.2f}s and frame: {currentFrame}")
                #error_text = f"Frozen Frame at {time_stamp:.2f}s and frame: {currentFrame}"
                #error_frame = frame.copy()
                #error_counter = error_duration
                frozen_frame_buffer.append(1)
            else:
                frozen_frame_buffer.append(0)

            # Only check the sum if the buffer has at least `window_size` elements
            if len(frozen_frame_buffer) >= window_size:
                #print(f"Window sum: {sum(frozen_frame_buffer)}")
                if sum(frozen_frame_buffer) > 4:
                    # print(f"Frozen Frame Error Detected at {round((currentFrame/self.fps), 4):.2f}s (More than 4 in the last {window_size} frames)")
                    # error_text = f"Frozen Frame Error at {round((currentFrame/self.fps), 4):.2f}s and frame: {currentFrame}"
                    if self.is_realtime:
                        self.error_text = f"Frozen Frame at {timestamp}"
                    else:
                        self.error_text = f"Frozen Frame Detected at {round((currentFrame/self.fps), 4)} seconds\nand frame: {currentFrame}"
                    self.update_log(self.error_text, "red")
                    self.error_frame = frame.copy()
                    self.error_counter = error_duration

                frozen_frame_buffer.pop(0)
                
                
        if ("general" in self.scripts and self.scripts["general"]):
            if hasattr(self,"prev_frame") and self.scripts["general"].general_detection(self.prev_frame,frame,sensitivity = 0.5-float(self.sensitivity.get())):
                if self.is_realtime:
                    self.error_text = f"High Probability of Error at {timestamp}"
                else:
                    self.error_text = f"High Probability of Error at {round((currentFrame/self.fps), 4)} seconds\nand frame: {currentFrame}"
                self.update_log(self.error_text, "red")
                self.error_frame = frame.copy()
                self.error_counter = error_duration
        # '''
        #if "mask" in self.scripts and "pano" in self.scripts and self.scripts["mask"] and self.scripts["pano"]:
        #    lmask, smask = self.scripts["mask"].create_mask(frame)
        #    if self.scripts["pano"].checkPano(frame, smask, lmask):
        #        error_text = f"Pano-70 Error Detected at {round((currentFrame/self.fps), 4)} seconds"
        #        self.update_log(error_text, "red")
        
        # Check pano autocorrelation
        if "pano" in self.scripts and self.scripts["pano"]:
            if self.scripts["pano"].repeated_region_numpy(frame):
                #error_text = f"Pano-70 (repeated region) Error at {round((currentFrame/self.fps), 4)}s and frame: {currentFrame}"
                if self.is_realtime:
                    self.error_text = f"Pano-70 Error at {timestamp}"
                else:
                    self.error_text = f"Pano-70 Error Detected at {round((currentFrame/self.fps), 4)} seconds\nand frame: {currentFrame}"
                # print(f"Pano-70 (repeated region) Error at {round((currentFrame/self.fps), 4)}s and frame: {currentFrame}")
                #error_frame = frame.copy()
                #error_counter = error_duration
                self.update_log(self.error_text, "red")
                self.error_frame = frame.copy()
                self.error_counter = error_duration

                
        # Calculate where to put the text
        frame_height, frame_width = frame.shape[:2]
        top_left = (int(0.02 * frame_width), int(0.05 * frame_height))
        center_pos = (int(0.4 * frame_width), int(0.5 * frame_height))
        error_pos = (int(0.02 * frame_width), int(0.15 * frame_height))

        # Create the error frames to be shown side by side with video
        if self.error_counter > 0:
            error_display = self.error_frame.copy()
            cv2.putText(error_display, 'Error Stream', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            lines = self.error_text.split('\n')
            line_spacing = 30  
            for i, line in enumerate(lines):
                position = (error_pos[0], error_pos[1] + i * line_spacing)
                cv2.putText(error_display, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            self.error_counter -= 1
        else:
            error_display = self.black_frame.copy()
            cv2.putText(error_display, 'Error Stream', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(error_display, 'No Error', center_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        error_display_frame = frame.copy()

        cv2.putText(error_display_frame, 'Input Video', top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Combine the frames side by side
        combined_frame = np.hstack((error_display_frame, error_display))

        # Write to Output Video
        self.error_vid.write(combined_frame)

        


        # Update previous frame for frozen frame detection
        self.prev_frame = frame

        # Display error frame in the canvas
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize frame to desired dimensions (width = 5/3 * height)
        frame = cv2.resize(frame, (640, 384))
        # Convert frame to PIL image
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Calculate the position to center the image on the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width = img_tk.width()
        img_height = img_tk.height()

        x = (canvas_width - img_width) // 2
        y = (canvas_height - img_height) // 2

        # Display the image on the canvas
        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=img_tk)
            self.canvas.coords(self.image_on_canvas, x, y)

        self.current_image = img_tk  # Store reference to avoid garbage collection

    def update_threshold(self):
        threshold = self.threshold_value.get()
        self.update_status(f"Threshold updated to {threshold}.", "N/A", "blue")
        # Update the threshold value in the error detection script here

def main():
    """root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()"""
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()