import sys
import os
import tkinter as tk
import keyboard
import subprocess
from datetime import datetime

# Add the error detection folder to sys.path by going back to the parent directory 
#parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
#sys.path.append(parent_dir)

'''
check everything is up to date
save videos? log?
txt file with file name + error timestamps and classification



'''



import importlib

# Function to load the correct error detection scripts dynamically
def load_error_detection_scripts(is_realtime=False):
    base_folder = "Consolidated - Real time - Arthroscope/source" if is_realtime else "Consolidated/source_update"
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
from tkinter import simpledialog, filedialog, ttk
from tkinter import Tk, Canvas, Text, Button, Label, Entry, StringVar, END
from skimage import measure
from skimage.draw import disk

def cv2_to_tk(img, scale=0.6):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    return ImageTk.PhotoImage(Image.fromarray(img))

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
window_size = 10
frozen_frame_buffer = []


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
            
            return self.shrunk_mask
        
        return self.shrunk_mask


class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")
        self.root.geometry("1000x660")
        self.source = 0

        self.lock = threading.Lock()
        self.is_realtime = False  # Flag to track real-time mode
        self.scripts = load_error_detection_scripts(self.is_realtime)

        # Create and place the canvas for video playback
        self.canvas = tk.Canvas(root, bg = "black" ,width=640, height=480)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        # Create and place the error log
        #self.error_log = tk.Text(root, height=30, width=40)
        self.error_log = tk.Text(root, height = 20)
        self.error_log.grid(row=0, column=3, sticky = 'ne', padx=5, pady=5)

        root.grid_columnconfigure(3, weight=2)

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

        self.autossh_button = tk.Button(root, text="Run AutoSSH", command=self.run_autossh)
        self.autossh_button.grid(row=3, column=2, padx=5, pady=5)

        self.open_logs_button = tk.Button(root, text="Open System Logs", command=self.open_system_logs)
        self.open_logs_button.grid(row=2, column=2, padx=5, pady=5)
        
        self.toggle_camera_button = tk.Button(root,text="Toggle Camera Source",command = self.toggle_camera)
        self.toggle_camera_button.grid(row=4,column=0,padx=5,pady=5)


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
        self.status_label.grid(row=1, column=3, padx=5, pady=5)
        
        self.footage_mask_label = tk.Label(root)
        self.footage_mask_label.grid(row=2, column=3, padx=0, pady=0)
    

        self.cap = None
        self.image_on_canvas = None
        self.play_flag = False
        self.file_path = None
        self.frozen_frame_flags = []
        self.start_time = None
        self.fps = 0

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)


    def run_autossh(self):
        try:
            ip = simpledialog.askstring("SSH IP", "Enter the remote IP address:", parent = self.root)
            username = simpledialog.askstring("SSH Username", "Enter the username:", parent = self.root)
            password = simpledialog.askstring("SSH Password", "Enter the password:", show="*", parent = self.root)

            if not all([ip, username, password]):
                self.status_label.config(text="SSH details incomplete.", fg="red")
                return

            # Pass the data to autossh.py
            subprocess.Popen(
                [sys.executable, "autossh.py", ip, username, password],
                shell=False
            )
            self.status_label.config(text="autossh.py launched with user input.", fg="green")

        except Exception as e:
            self.status_label.config(text=f"Failed to run autossh.py: {e}", fg="red")

    def open_system_logs(self):
        try:
            with open("arthrex_system_logs.txt", "r") as file:
                content = file.read()

            # Create a new popup window
            log_window = tk.Toplevel(self.root)
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
        
        self.source = (self.source + 1) % len(sources)
        self.cap = cv2.VideoCapture(self.source)


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

    def save_log_to_file(self, filename="error_log.txt"):
        with open(filename, 'w') as file:
            file.write(self.error_log.get("1.0", tk.END))

    def reset_gui(self):
        '''
        """Reset the entire window by recreating the main window."""
        self.root.quit()  # Close the current window
        new_root = tk.Tk()  # Create a new Tkinter window
        new_window = VideoPlayer(new_root)  # Reinitialize the VideoPlayer window
        new_root.mainloop()  # Start the new event loop
        '''

        # Stop any playback or video capture
        self.play_flag = False
        if self.cap:
            self.cap.release()
            self.cap = None

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

            file_name = os.path.basename(self.file_path)  # Extract just the file name
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
        
        self.scripts = load_error_detection_scripts(self.is_realtime)
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



                # Update the progress bar and check if the video is playing in real time
                current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                
                # Make sure toggling the camera does not mess up the post-processing timestamp code
                try:
                    total_time = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
                
                except ZeroDivisionError:
                    if self.cap is not None:
                        self.cap.release()
                    self.source = 0
                    self.cap = cv2.VideoCapture(self.source)
                    total_time = 1
                    
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

                if not self.is_realtime:
                    # After processing is done
                    file_name = self.file_path # File path included
                    # file_name = os.path.basename(self.file_path)  # Extract just the file name
                    self.update_log(f"Video file '{file_name}' is done being processed...", "blue")  # Add log update
                    self.save_log_to_file()





    def detect_and_display_errors(self, frame):
        
        '''
        Masking the entire footage is unnecessary, we only mask for dropout
        
        if hasattr(self, 'current_mask') and self.current_mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=self.current_mask)
            mask_applied = True
        else:
            mask_applied = False
        '''
        mask_applied = False

        with self.lock:
            error_text = "No Error"
            currentFrame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # e.g., 2025-05-05 14:32:21.123

        # Use dynamically loaded modules
        if "green" in self.scripts and self.scripts["green"]:
            
            #if self.scripts["green"].checkGreenFrame_numba(frame) == 1:
            #    error_text = "Green Screen Error" if self.is_realtime else f"Green Screen Error Detected at {round((currentFrame/self.fps), 4)} seconds and frame: {currentFrame}"
            #    self.update_log(error_text, "red")

            if self.scripts["green"].checkGreenFrame_numba(frame) == 1:
                if self.is_realtime:
                    error_text = f"Full Green Screen Error at {timestamp}"
                else:
                    error_text = f"Full Green Screen Error Detected at {round((currentFrame/self.fps), 4)} seconds and frame: {currentFrame}"
                self.update_log(error_text, "red")


            if self.scripts["green"].checkGreenFrame_numba(frame) == 2:
                if self.is_realtime:
                    error_text = f"Corner Green Screen Error at {timestamp}"
                else:
                    error_text = f"Corner Green Screen Error Detected at {round((currentFrame/self.fps), 4)} seconds and frame: {currentFrame}"
                self.update_log(error_text, "red")

        if "magenta" in self.scripts and self.scripts["magenta"]:
            if self.scripts["magenta"].checkMagentaFrame_numba(frame):
                if self.is_realtime:
                    error_text = f"Magenta Screen Error at {timestamp}"
                else:
                    error_text = f"Magenta Screen Error Detected at {round((currentFrame/self.fps), 4)} seconds and frame: {currentFrame}"
                self.update_log(error_text, "red")

        if "black" in self.scripts and self.scripts["black"]:
            if self.masking.raw_mode:
                if self.scripts["black"].checkDropoutNoMask(frame):
                    if self.is_realtime:
                        error_text = f"Dropout Error at {timestamp}"
                    else:
                        error_text = f"Dropout Error Detected at {round((currentFrame/self.fps), 4)} seconds and frame: {currentFrame}"
                    self.update_log(error_text, "red")
            else:
                if self.scripts["black"].checkBlackFrame_numba(frame, self.current_mask):
                    if self.is_realtime:
                        error_text = f"Dropout Error at {timestamp}"
                    else:
                        error_text = f"Dropout Error Detected at {round((currentFrame/self.fps), 4)} seconds and frame: {currentFrame}"
                    self.update_log(error_text, "red")

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
                        error_text = f"Frozen Frame at {timestamp}"
                    else:
                        error_text = f"Frozen Frame Detected at {round((currentFrame/self.fps), 4)} seconds and frame: {currentFrame}"
                    self.update_log(error_text, "red")
                    #error_frame = frame.copy()
                    #error_counter = error_duration

                frozen_frame_buffer.pop(0)
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
                    error_text = f"Pano-70 Error at {timestamp}"
                else:
                    error_text = f"Pano-70 Error Detected at {round((currentFrame/self.fps), 4)} seconds and frame: {currentFrame}"
                # print(f"Pano-70 (repeated region) Error at {round((currentFrame/self.fps), 4)}s and frame: {currentFrame}")
                #error_frame = frame.copy()
                #error_counter = error_duration
                self.update_log(error_text, "red")

        


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
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()

if __name__ == "__main__":
    main()