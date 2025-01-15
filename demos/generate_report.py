import os
import subprocess

def generate_report(configs_folder, ath_exe_path, config_filename, verbose=False):
    """
    Generates a report for the given configuration file by running the ATH executable.

    Parameters:
    - config_filename (str): Name of the configuration file.
    - configs_folder (str): Path to the folder containing configuration files.
    - ath_exe_path (str): Path to the ATH executable.
    - verbose (bool): If True, print additional debug information.

    Returns:
    - True if the report was generated successfully, False otherwise.
    """
    try:
        # Construct the full path to the config file
        config_path = os.path.join(configs_folder, config_filename)
        command = [ath_exe_path, config_path, "-r"]

        # Run the command and capture output
        if verbose:
            print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        # Check the result
        if result.returncode == 0:
            if verbose:
                print(f"Success: {config_filename}")
            return True
        else:
            if verbose:
                print(f"Error running {config_filename}: {result.stderr}")
            return False
    except Exception as e:
        if verbose:
            print(f"Unexpected error while running the report: {e}")
        return False
