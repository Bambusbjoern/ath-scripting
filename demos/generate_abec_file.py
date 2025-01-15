import os
import subprocess

def generate_abec_file(configs_folder, ath_exe_path, config_filename, verbose=False):
    """
    Runs ATH simulation for a specified configuration file within the given configs folder.

    Parameters:
    - configs_folder (str): Path to the folder containing configuration files.
    - ath_exe_path (str): Path to the ATH executable.
    - config_filename (str): Name of the configuration (.cfg) file.
    - verbose (bool): If True, print additional debug information.

    Returns:
    - True if the simulation ran successfully, False otherwise.
    """
    # Construct the full path to the config file
    config_file_path = os.path.join(configs_folder, config_filename)
    
    # Construct the command to run ath.exe with the config file
    command = [ath_exe_path, config_file_path]

    # Run the command and capture output
    if verbose:
        print(f"Running command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            if verbose:
                print(f"Success: {config_filename}")
                print(result.stdout)
            return True  # Indicate successful run
        else:
            if verbose:
                print(f"Error running {config_filename}: {result.stderr}")
            return False  # Indicate failure

    except Exception as e:
        if verbose:
            print(f"Exception occurred while running command: {e}")
        return False
