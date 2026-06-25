# for each xml file in the input folder, launch the corresponding application
import os
import sys
import random
import subprocess
import cv2
import numpy as np
import mss
import time
import platform
import pyautogui
import shutil

def get_window_geometry(pid):
    """
    Helper: Returns {'top', 'left', 'width', 'height'} for a PID.
    Supports Windows, Linux (requires xdotool), and macOS.
    """
    system = platform.system()

    if system == "Windows":
        try:
            import win32gui
            import win32process
        except ImportError:
            print("Please install pywin32: pip install pywin32")
            return None

        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                if found_pid == pid:
                    rect = win32gui.GetWindowRect(hwnd)
                    windows.append({
                        "left": rect[0], "top": rect[1],
                        "width": rect[2] - rect[0], "height": rect[3] - rect[1]
                    })
        windows = []
        win32gui.EnumWindows(callback, windows)
        return max(windows, key=lambda x: x['width'] * x['height']) if windows else None

    elif system == "Linux":
        try:
            wid_bytes = subprocess.check_output(["xdotool", "search", "--pid", str(pid), "--onlyvisible"])
            wid = wid_bytes.decode().strip().split('\n')[-1]
            geo_output = subprocess.check_output(["xdotool", "getwindowgeometry", wid]).decode()
            import re
            pos = re.search(r"Position: (\d+),(\d+)", geo_output)
            geo = re.search(r"Geometry: (\d+)x(\d+)", geo_output)
            if pos and geo:
                return {
                    "left": int(pos.group(1)), "top": int(pos.group(2)),
                    "width": int(geo.group(1)), "height": int(geo.group(2))
                }
        except Exception:
            return None
    return None

def launch_and_record(command, output_filename, duration=10, fps=45.0):
    """
    Launches a program, finds its window, and records it.
    """
    import os # Ensure os is imported locally or globally
    
    print(f"--- Starting: {command[0]} ---")
    
    # 1. Determine the directory of the executable
    exe_path = command[0]
    # Get the folder containing the exe (e.g., C:\Program Files (x86)\torcs\)
    working_dir = os.path.dirname(exe_path)
    
    # If the command was just "wtorcs.exe" (no path), working_dir might be empty.
    # If so, we don't change cwd.
    if not working_dir: 
        working_dir = None
    else:
        print(f"Setting working directory to: {working_dir}")

    # 2. Launch App with 'cwd' argument
    try:
        # cwd tells the OS "pretend I am in this folder when you start"
        proc = subprocess.Popen(command, cwd=working_dir)
    except FileNotFoundError:
        print(f"Error: Could not find program '{command[0]}'")
        return
    # --- FIX ENDS HERE ---

    # 3. Wait for Window (up to 10 seconds)
    print(f"Waiting for window (PID: {proc.pid})...")
    coords = None
    for _ in range(10):
        coords = get_window_geometry(proc.pid)
        if coords: break
        time.sleep(1)

    # Sequence: Type 'user', hit Tab, type 'password', hit Enter
    keys_sequence = ['enter', 'enter','enter','enter']

    # This will press each key in the list with a 100ms delay in between
    pyautogui.write(keys_sequence, interval=0.1)
    
    if not coords:
        print("Window not found. Killing process.")
        # Try to terminate gracefully, then force kill if needed
        proc.terminate()
        return

    print(f"Window found: {coords}. Recording to {output_filename}...")

    # 4. Setup Recorder
    # Use mp4v or XVID. mp4v is generally safer for standard MP4s.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (coords['width'], coords['height']))
    
    start_time = time.time()
    specIsForced = False
    
    with mss.mss() as sct:
        try:
            while True:
                # Check if process is still running
                if proc.poll() is not None:
                    print("Application closed naturally.")
                    break
                
                # Check duration
                if duration > 0 and (time.time() - start_time) > duration:
                    print("Duration reached.")
                    break
                
                # after 10 seconds try to force spec cam with f7
                if (not specIsForced and time.time() - start_time) > 5:
                    # sequenceof f11 + home
                    keys_sequence = ['f11', 'home']
                    pyautogui.write(keys_sequence, interval=0.1)
                    specIsForced = True
                # Capture Frame
                try:
                    img = sct.grab(coords)
                    frame = np.array(img)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    out.write(frame)
                except Exception as e:
                    print(f"Screen capture failed: {e}")
                    break
                    
        except KeyboardInterrupt:
            print("Recording stopped manually.")

    # 5. Cleanup
    out.release()
    print(f"Finished. Saved to {output_filename}\n")
    
    # Kill the app if it's still running
    if proc.poll() is None:
        proc.terminate()
# ==========================================



xmlDirectory = "data/Archive/voronoi final mixed with new bot tita/xmlTracks/"
videoDirectory = "data/videos/"
ps_script_path  = "src/utils/generateoutput.ps1"

# choose one randomly from the tracks in the xmlTracks folder
xmlFiles = [
        f for f in os.listdir(xmlDirectory) if f.endswith('.xml')
    ]

for i in range(25):
  if not xmlFiles:
    sys.exit("ERROR: No XML track files found!")
  randomFile = random.choice(xmlFiles)

  cmd = [
    "powershell.exe",
    "-ExecutionPolicy", "Bypass",
    "-File", ps_script_path,
    randomFile  # This passes the argument to the param() block in PowerShell
  ]
  try:
      print(f"Running PowerShell script for {randomFile}...")
      result = subprocess.run(cmd, capture_output=False, text=True, check=True)
      
      # Print the output from PowerShell
      print("STDOUT:", result.stdout)

  except subprocess.CalledProcessError as e:
      print("Error occurred!")
      print("STDERR:", e.stderr)

  launch_and_record(
        command=["C:\\Program Files (x86)\\torcs\\wtorcs.exe"], 
        output_filename= videoDirectory + randomFile.replace('.xml', '.mp4'), 
        duration=300
    )  
