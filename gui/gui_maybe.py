import sys
import os
import tkinter as tk

# Add the error detection folder to sys.path by going back to the parent directory 
#parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
#sys.path.append(parent_dir)

import importlib

# Function to load the correct error detection scripts dynamically
def load_error_detection_scripts(is_realtime=False):
    base_folder = "Consolidated - Real time - Arthroscope/source" if is_realtime else "Consolidated/source"
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
    }

    loaded_scripts = {}
    for key, module_name in modules.items():
        try:
            loaded_scripts[key] = importlib.import_module(module_name)
        except ModuleNotFoundError:
            print(f"Warning: {module_name}.py not found in {folder_path}")

    return loaded_scripts


import threading
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tkinter import filedialog, ttk
from tkinter import Tk, Canvas, Text, Button, Label, Entry, StringVar, END

# Import the error detection scripts
'''
from greenScripts.greenVectorizedSolution import checkGreenFrame
from greenScripts.magentaScreen import checkMagentaFrame
from greenScripts.dropoutScreen import checkBlackFrame
from Highlights.highlights import checkHighlightsFrame
from Frozen.lagff15 import detect_frozen_frame
from HelperScripts.auto_mask import create_mask
from panoto70.panoto70fcn import checkPano
'''

currentFrame = 1

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")
        self.root.geometry("1000x660")

        self.lock = threading.Lock()
        self.is_realtime = False  # Flag to track real-time mode
        self.scripts = load_error_detection_scripts(self.is_realtime)

        # Create and place the canvas for video playback
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        # Create and place the error log
        self.error_log = tk.Text(root, height=30, width=40)
        self.error_log.grid(row=0, column=3, padx=5, pady=5)

        # Create and place the video progress bar
        self.progress_label = tk.Label(root, text="0:00 / 0:00")
        self.progress_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=640, mode='determinate')
        self.progress_bar.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        # Create and place the Choose File button
        self.choose_button = tk.Button(root, text="Choose File", command=self.choose_file)
        self.choose_button.grid(row=2, column=0, padx=5, pady=5)

        # Create and place the Real-Time button
        self.realtime_button = tk.Button(root, text="Real-Time", command=self.realtime_video)
        self.realtime_button.grid(row=3, column=0, padx=5, pady=5)

        # Create and place the Play button
        self.play_button = tk.Button(root, text="Play", command=self.play_video)
        self.play_button.grid(row=2, column=1, padx=5, pady=5)
        self.play_button.config(state=tk.DISABLED)

        # Reset button
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_gui)
        self.reset_button.grid(row=3, column=1, padx=5, pady=5)

        # Create and place the threshold controls
        #self.threshold_label = tk.Label(root, text="Green Detection Threshold:")
        #self.threshold_label.grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        #self.threshold_value = tk.StringVar(value="100")  # Default threshold value
        #self.threshold_entry = tk.Entry(root, textvariable=self.threshold_value)
        #self.threshold_entry.grid(row=2, column=2, padx=5, pady=5)
        #self.update_threshold_button = tk.Button(root, text="Update Threshold", command=self.update_threshold)
        #self.update_threshold_button.grid(row=2, column=2, padx=5, pady=5, sticky=tk.E)

        # Create and place the status label
        self.status_label = tk.Label(root, text="No video detected.\nRemaining time: N/A", fg="red")
        self.status_label.grid(row=2, column=3, padx=5, pady=5)

        self.cap = None
        self.image_on_canvas = None
        self.play_flag = False
        self.file_path = None
        self.frozen_frame_flags = []
        self.start_time = None
        self.fps = 0

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)


    def on_close(self):
        if self.cap is not None and self.cap.isOpened():  # Check if video capture exists
            self.cap.release()  # Release the camera or video
            print("Camera released")
    
        cv2.destroyAllWindows()  # Close OpenCV windows
        self.root.destroy()  # Close Tkinter window

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

    def reset_gui(self):
        """Reset the entire window by recreating the main window."""
        self.root.quit()  # Close the current window
        new_root = tk.Tk()  # Create a new Tkinter window
        new_window = VideoPlayer(new_root)  # Reinitialize the VideoPlayer window
        new_root.mainloop()  # Start the new event loop

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
            self.scripts = load_error_detection_scripts(self.is_realtime)

            # Release previous capture if it exists
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.cap = None  # Reset the cap to avoid stale reference

            self.cap = cv2.VideoCapture(self.file_path)
            
            if not self.cap.isOpened():
                self.update_status("Error with file.", "N/A", "red")
                return

            self.start_time = time.time()  # Post-processing does not need real-time tracking
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            file_name = os.path.basename(self.file_path)  # Extract just the file name
            self.update_status("Video is currently playing.", "Calculating...", "green")
            self.update_log(f"Video file '{file_name}' is now being processed...", "blue")  # Add log update
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
        self.scripts = load_error_detection_scripts(self.is_realtime)

        # Clear the canvas
        self.canvas.delete("all")
        self.image_on_canvas = None  # Reset canvas display
        #print("DEBUG: canvas cleared")
        
        # Release previous capture if it exists
        if self.cap is not None:
            #print("DEBUG: releasing previous video capture")
            self.cap.release()
            self.cap = None  # Reset to avoid stale reference
        
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.cap = None
            self.update_status("Error with webcam.", "N/A", "red")
            return
        
        self.is_realtime = True  # Set mode to real-time

        self.play_flag = True  # Make sure this is set before looping
        self.start_time = time.time()
        self.fps = 30  # Approximate real-time FPS
        #print("DEBUG: webcam opened successfully")
        self.update_status("Real-time video is currently playing.", "green")
        
        self.process_video_frame()

    def process_video_frame(self):
        #print("DEBUG: processing video frame")
        if self.play_flag and self.cap is not None:
            ret, frame = self.cap.read()
            # print(f" frame retrieved: {ret}")
            
            if ret:
                #print("DEBUG: frame read successfully")
                frame = cv2.resize(frame, (640, 480))
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
                    self.update_status(f"Video playing in real time.")
                else:
                    if abs(elapsed_real_time - current_time) < 1:
                        self.update_status(f"Video playing in real time. ({percentage:.2f}%)", "N/A", "green")
                    else:
                        remaining_real_time = (total_time - current_time) / speed_ratio
                        self.update_status(f"Video not playing in real time. ({percentage:.2f}%)", f"{int(remaining_real_time//60)}:{int(remaining_real_time%60):02d}", "red")

                self.root.after(5, self.process_video_frame)
                #self.root.after(16, self.process_video_frame)

            else:
                #print("DEBUG: no frame captured")
                self.cap.release()
                self.update_status("Finished playing video.", "0:00", "blue")
                self.canvas.delete("all")





    def detect_and_display_errors(self, frame):
        with self.lock:
            error_text = "No Error"
            currentFrame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Use dynamically loaded modules
        if "green" in self.scripts and self.scripts["green"]:
            if self.scripts["green"].checkGreenFrame(frame):
                error_text = "Green Screen Error" if self.is_realtime else f"Green Screen Error Detected at {round((currentFrame/self.fps), 4)} seconds"
                self.update_log(error_text, "red")

        if "magenta" in self.scripts and self.scripts["magenta"]:
            if self.scripts["magenta"].checkMagentaFrame(frame):
                error_text = "Magenta Screen Error" if self.is_realtime else f"Magenta Screen Error Detected at {round((currentFrame/self.fps), 4)} seconds"
                self.update_log(error_text, "red")

        if "black" in self.scripts and self.scripts["black"]:
            if self.scripts["black"].checkBlackFrame(frame):
                error_text = "Dropout Error" if self.is_realtime else f"Dropout Error Detected at {round((currentFrame/self.fps), 4)} seconds"
                self.update_log(error_text, "red")

        #if "highlights" in self.scripts and self.scripts["highlights"]:
        #    if self.scripts["highlights"].checkHighlightsFrame(frame):
        #        error_text = f"Highlight Shimmer Detected at {round((currentFrame/self.fps), 4)} seconds"
        #        self.update_log(error_text, "red")

        if "frozen" in self.scripts and self.scripts["frozen"]:
            if hasattr(self, "prev_frame") and self.scripts["frozen"].detect_frozen_frame(self.prev_frame, frame):
                error_text = "Frozen Frame" if self.is_realtime else f"Frozen Frame Detected at {round((currentFrame/self.fps), 4)} seconds"
                self.update_log(error_text, "red")

        #if "mask" in self.scripts and "pano" in self.scripts and self.scripts["mask"] and self.scripts["pano"]:
        #    lmask, smask = self.scripts["mask"].create_mask(frame)
        #    if self.scripts["pano"].checkPano(frame, smask, lmask):
        #        error_text = f"Pano-70 Error Detected at {round((currentFrame/self.fps), 4)} seconds"
        #        self.update_log(error_text, "red")


        # Update previous frame for frozen frame detection
        self.prev_frame = frame

        # Display error frame in the canvas
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=img_tk)

        self.current_image = img_tk  # Store reference to avoid garbage collection

    def update_threshold(self):
        threshold = self.threshold_value.get()
        self.update_status(f"Threshold updated to {threshold}.", "N/A", "blue")
        # Update the threshold value in the error detection script here

def main():
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
