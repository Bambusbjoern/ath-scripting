import os
import subprocess
import time
import pygetwindow as gw
import pyautogui
import math
import configparser
import tkinter as tk

# Load configurations from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Paths from config.ini
HORNS_FOLDER = config['Paths']['HORNS_FOLDER']
ABEC_EXE_PATH = config['Paths']['ABEC_EXE_PATH']

# Window settings from config.ini
WINDOW_WIDTH = config.getint('WindowSettings', 'WINDOW_WIDTH')
WINDOW_HEIGHT = config.getint('WindowSettings', 'WINDOW_HEIGHT')
WINDOW_X = config.getint('WindowSettings', 'WINDOW_X')
WINDOW_Y = config.getint('WindowSettings', 'WINDOW_Y')

# Pixel check settings from config.ini
PIXEL_X = config.getint('PixelCheck', 'PIXEL_X')
PIXEL_Y = config.getint('PixelCheck', 'PIXEL_Y')
EXPECTED_COLOR = tuple(map(int, config['PixelCheck']['EXPECTED_COLOR'].split(',')))
COLOR_THRESHOLD = config.getint('PixelCheck', 'COLOR_THRESHOLD')
REF_PIXEL_X = config.getint('PixelCheck', 'REF_PIXEL_X')
REF_PIXEL_Y = config.getint('PixelCheck', 'REF_PIXEL_Y')
REF_COLOR = tuple(map(int, config['PixelCheck']['REF_COLOR'].split(',')))

# Retries settings from config.ini
MAX_RETRIES = config.getint('Retries', 'MAX_RETRIES')
TIMEOUT = config.getint('Retries', 'TIMEOUT')
CALCULATION_TIMEOUT = config.getint('Retries', 'CALCULATION_TIMEOUT')

import sys

def get_simulation_folder_type(template_path="base_template.txt"):
    try:
        with open(template_path, "r") as file:
            for line in file:
                # Remove any comment after the `;` character
                line = line.split(";")[0].strip()
                
                # Look for the line defining `ABEC.SimType`
                if line.startswith("ABEC.SimType"):
                    sim_type = int(line.split("=")[1].strip())
                    return "ABEC_InfiniteBaffle" if sim_type == 1 else "ABEC_FreeStanding"
                    
        # If we complete the loop and don't find ABEC.SimType, raise an error
        raise ValueError("ABEC.SimType not found in the template.")
    
    except FileNotFoundError:
        print(f"Template file not found: {template_path}")
        sys.exit("Exiting: Please ensure the template file is available.")
    except ValueError as e:
        print(f"Error reading simulation type from template: {e}")
        sys.exit("Exiting: Please check the format of ABEC.SimType in the template file.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit("Exiting: An unexpected error occurred.")


# Helper function to calculate the Euclidean distance between two colors (RGB)
def color_distance(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

# Start the ABEC program with the file
def start_abec_with_file(program_path, file_path):
    command = f'"{program_path}" "{file_path}"'
    print(f"Starting: {command}")
    subprocess.Popen(command, shell=True)

# Wait until the window is available by checking the start of the window title
def wait_for_window(partial_title, timeout=10):
    print(f"Waiting for a window starting with: {partial_title}")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        windows = gw.getAllTitles()
        for title in windows:
            if title.startswith(partial_title):
                print(f"Window found: {title}")
                return gw.getWindowsWithTitle(title)[0]
        time.sleep(0.1)
    
    print(f"Timeout: No window starting with '{partial_title}' found")
    return None


def wait_for_window_disappearance(partial_title, timeout=10):
    print(f"Waiting for the window with title '{partial_title}' to close...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        windows = gw.getAllTitles()
        if not any(title.startswith(partial_title) for title in windows):
            print(f"Window with title '{partial_title}' has been closed.")
            return True
        time.sleep(0.1)

    # If timeout is reached and the window still exists
    print(f"Timeout: Window with title '{partial_title}' did not close within {timeout} seconds. Retrying to close it...")
    
    # Try to refocus and close the window with Alt+F4 and Enter
    try:
        # Get the window with the given title
        abec_window = gw.getWindowsWithTitle(partial_title)[0]
        
        # Focus the window
        abec_window.activate()
        time.sleep(1)  # Give a brief moment to refocus
        
        # Send Alt+F4 and Enter to close it
        pyautogui.hotkey('alt', 'f4')
        time.sleep(0.5)
        pyautogui.press('enter')
        
        # Wait briefly to check if the window closes
        time.sleep(3)
        
        # Re-check if the window has closed
        if not any(title.startswith(partial_title) for title in gw.getAllTitles()):
            print(f"Window with title '{partial_title}' has been closed successfully.")
            return True
        else:
            print(f"Failed to close the window with title '{partial_title}'.")
            return False
    
    except IndexError:
        print(f"No window with title '{partial_title}' was found to close.")
        return False


# Resize and move the window
def set_window_size_and_position(window):
    window.resizeTo(WINDOW_WIDTH, WINDOW_HEIGHT)
    window.moveTo(WINDOW_X, WINDOW_Y)
    pyautogui.moveTo((WINDOW_X + (WINDOW_WIDTH // 2)), (WINDOW_Y + (WINDOW_HEIGHT // 2)))
    print(f"Window resized to {WINDOW_WIDTH}x{WINDOW_HEIGHT} and moved to ({WINDOW_X}, {WINDOW_Y})")

# Start the solver by pressing F5 followed by Enter
def start_solver():
    print("Starting solver with F5 and Enter...")
    pyautogui.press('f5')
    time.sleep(0.25)
    pyautogui.press('enter')

# Check the color of the reference pixel and apply the initial timeout
def wait_for_reference_pixel(timeout=TIMEOUT):
    print(f"Waiting for the reference pixel at ({REF_PIXEL_X}, {REF_PIXEL_Y}) to match the expected color: {REF_COLOR}")
    start_time = time.time()
    previous_color = None  # Track the last color value

    while time.time() - start_time < timeout:
        pixel_color = pyautogui.pixel(REF_PIXEL_X, REF_PIXEL_Y)
        distance = color_distance(pixel_color, REF_COLOR)
        
        # Print only if color changes
        if pixel_color != previous_color:
            print(f"Reference pixel color: {pixel_color}, distance from expected: {distance:.2f}")
            previous_color = pixel_color  # Update the tracked color
        
        if distance <= COLOR_THRESHOLD:
            print("Reference pixel color is close enough to the expected value.")
            return True
        time.sleep(0.25)
    
    print(f"Timeout: Reference pixel color did not match the expected value within {timeout} seconds.")
    return False


# Monitor calculation progress with the calculation timeout
def monitor_calculation_progress(calculation_timeout=CALCULATION_TIMEOUT, abec_window_title="ABEC3"):
    print(f"Monitoring calculation progress for up to {calculation_timeout} seconds at ({PIXEL_X}, {PIXEL_Y}) for the expected color: {EXPECTED_COLOR}")
    start_time = time.time()
    previous_color = None  # Track the last color value

    while time.time() - start_time < calculation_timeout:
        # Check if the ABEC window still exists
        if not any(title.startswith(abec_window_title) for title in gw.getAllTitles()):
            print("ABEC window not found. It may have closed unexpectedly.")
            return False  # Stop monitoring if window is gone

        # Check the color of the calculation progress pixel
        pixel_color = pyautogui.pixel(PIXEL_X, PIXEL_Y)
        distance = color_distance(pixel_color, EXPECTED_COLOR)
        
        # Print only if color changes
        if pixel_color != previous_color:
            print(f"Calculation progress pixel color: {pixel_color}, distance from expected: {distance:.2f}")
            previous_color = pixel_color  # Update the tracked color
        
        if distance <= COLOR_THRESHOLD:
            print("Calculation progress pixel color is close enough to the expected value.")
            return True
        time.sleep(0.5)
    
    print(f"Calculation timeout reached: Did not detect expected color within {calculation_timeout} seconds.")
    return False


# Retry logic for solving with up to three attempts
def solver_with_retry(abec_window):
    retry_step = 0

    while retry_step < MAX_RETRIES:
        if wait_for_reference_pixel(TIMEOUT):
            return monitor_calculation_progress()

        retry_step += 1
        if retry_step == 1:
            print("Timeout reached again, resizing and repositioning window.")
            set_window_size_and_position(abec_window)
        elif retry_step == 2:
            print("Timeout reached, restarting solver.")
            start_solver()
        elif retry_step == MAX_RETRIES:
            print("Timeout reached a third time. Quitting ABEC.")
            pyautogui.hotkey('alt', 'f4')
            time.sleep(0.25)
            pyautogui.press('enter')
            wait_for_window_disappearance('ABEC3')  # Ensure ABEC is closed before continuing
            return False
    return False

# Calculate spectra after verifying progress pixel color
def calculate_spectra():
    time.sleep(15)
    print("Pressing F7 to calculate spectra...")
    pyautogui.press('f7')

    # Wait for the progress pixel to indicate calculation progress
    if monitor_calculation_progress():
        time.sleep(2.5)
        print("Pressing Ctrl + F7 to finalize spectra...")
        pyautogui.hotkey('ctrl', 'f7')
        time.sleep(1.5)
        print("Pressing Enter to confirm the popup...")
        pyautogui.press('enter')
        time.sleep(3)
        print("Closing the program with Alt + F4...")
        pyautogui.hotkey('alt', 'f4')
        time.sleep(1.25)
        pyautogui.press('enter')
        wait_for_window_disappearance('ABEC3')

# Define tooltip function
def show_tooltip(text):
    tooltip = tk.Tk()
    tooltip.overrideredirect(True)
    tooltip.attributes("-topmost", True)
    tooltip.geometry("+10+10")  # Top-left corner of the screen
    label = tk.Label(tooltip, text=text, background="lightyellow", font=("Arial", 10))
    label.pack()
    tooltip.update()  # Display immediately
    return tooltip

def run_abec_simulation(folder_name):
    simulation_folder = get_simulation_folder_type("base_template.txt")
    folder_path = os.path.join(HORNS_FOLDER, folder_name, simulation_folder)
    project_file_path = os.path.join(folder_path, "Project.abec")
    
    # Check if Project.abec file exists in the specified folder
    if not os.path.exists(project_file_path):
        print(f"Project.abec file not found in {folder_path}")
        return False

    # Start ABEC with the project file
    print(f"Processing: {project_file_path}")
    start_abec_with_file(ABEC_EXE_PATH, project_file_path)

    # Wait for ABEC window to appear
    abec_window = wait_for_window('ABEC3')
    if abec_window:
        set_window_size_and_position(abec_window)
        time.sleep(1.5)
        start_solver()

        # Retry mechanism for solving, including monitoring progress
        if not solver_with_retry(abec_window):
            print(f"Failed to complete process for {project_file_path}")
            return False
        else:
            calculate_spectra()
            print(f"Process completed successfully for {project_file_path}")
            return True
    else:
        print("Failed to find ABEC window.")
        return False
