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

        # Ensure working directory is set to the executable's folder
        cwd = os.path.dirname(ath_exe_path)

        if verbose:
            print(f"ATH executable path: {ath_exe_path}")
            print(f"Configuration file path: {config_path}")
            print(f"Command to execute: {' '.join(command)}")
            print(f"Working directory: {cwd}")

        # Run the command
        env = os.environ.copy()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env  # Pass environment variables
        )

        # Check the result
        if result.returncode == 0:
            if verbose:
                print(f"Success: {config_filename}")
            return True
        else:
            if verbose:
                print(f"Command failed with return code: {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        if verbose:
            print(f"Unexpected error while running the report: {e}")
        return False
