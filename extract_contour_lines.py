import os
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, interp1d
import configparser

# Load configurations from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
horns_folder = config['Paths']['HORNS_FOLDER']

# Define the contour levels for which angles will be extracted
contour_levels = [-10, -6, -3, -2, -1]

# Define 500 log-spaced frequency points from 200 Hz to 20,000 Hz for CSV export
export_frequencies = np.logspace(np.log10(200), np.log10(20000), 500)

# Function to determine simulation folder based on ABEC.SimType in base_template.txt
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


def extract_contour_lines(folder_name):
    # Determine the correct simulation folder based on ABEC.SimType
    simulation_folder = get_simulation_folder_type("base_template.txt")
    
    # Construct the full path to polars.txt within the specified folder path
    folder_path = os.path.join(horns_folder, folder_name, simulation_folder, "Results")
    file_path = os.path.join(folder_path, "polars.txt")

    # Check if polars.txt exists at the constructed file path
    if not os.path.isfile(file_path):
        print(f"polars.txt not found in {file_path}, skipping...")
        return False

    # Load the data
    data = pd.read_csv(file_path, sep=' ', header=None, names=["Frequency", "Level_dB", "Angle_Index"])

    # Define the angles based on the angle index
    angle_step = 90 / 18  # 19 points for 0° to 90°
    angles = np.arange(0, 91, angle_step)
    data["Angle"] = data["Angle_Index"].apply(lambda x: angles[int(x) - 1])

    # Prepare the data for interpolation
    frequencies = data["Frequency"].to_numpy()
    angles = data["Angle"].to_numpy()
    levels = data["Level_dB"].to_numpy()

    # Create a grid for interpolation over frequencies and angles
    grid_freqs, grid_angles = np.meshgrid(
        np.logspace(np.log10(frequencies.min()), np.log10(frequencies.max()), 300),
        np.linspace(0, 90, 100)
    )

    # Interpolate SPL levels on the grid
    grid_levels = griddata(
        (np.log10(frequencies), angles), levels,
        (np.log10(grid_freqs), grid_angles),
        method='cubic'
    )

    # Dictionary to store the results for CSV, rounding to two decimals
    extracted_contours = {"Frequency": np.round(export_frequencies, 2)}

    # Extract and interpolate angles for each contour level
    for level in contour_levels:
        angle_column = []
        for freq in export_frequencies:
            # Find the closest index in grid_freqs to the current frequency
            freq_idx = (np.abs(grid_freqs[0] - freq)).argmin()

            # Calculate the difference between SPL values and the contour level
            level_diff = grid_levels[:, freq_idx] - level

            # Identify where the level crosses the contour level
            angle_indices = np.where(np.diff(np.sign(level_diff)))[0]

            if angle_indices.size > 1:
                # Extract nearby angles and levels for interpolation
                angles_nearby = grid_angles[angle_indices, 0]
                levels_nearby = grid_levels[angle_indices, freq_idx]

                try:
                    # Try cubic interpolation
                    cubic_interp = interp1d(levels_nearby, angles_nearby, kind='cubic', bounds_error=False, fill_value="extrapolate")
                    crossing_angle = cubic_interp(level)
                except ValueError:
                    # If cubic interpolation fails, fallback to linear interpolation
                    linear_interp = interp1d(levels_nearby, angles_nearby, kind='linear', bounds_error=False, fill_value="extrapolate")
                    crossing_angle = linear_interp(level)

                # Extract single value and round to two decimals
                angle_column.append(round(float(crossing_angle), 2) if np.ndim(crossing_angle) == 0 else round(float(crossing_angle[0]), 2))

            elif angle_indices.size == 1:
                # Fallback to linear interpolation if only one crossing is found
                lower_idx, upper_idx = angle_indices[0], angle_indices[0] + 1
                lower_angle, upper_angle = grid_angles[lower_idx, 0], grid_angles[upper_idx, 0]
                lower_level, upper_level = grid_levels[lower_idx, freq_idx], grid_levels[upper_idx, freq_idx]

                crossing_angle = lower_angle + (level - lower_level) * (upper_angle - lower_angle) / (upper_level - lower_level)
                angle_column.append(round(crossing_angle, 2))

            else:
                # If no crossing found, append NaN
                angle_column.append(np.nan)

        # Add the column for the current contour level to the dictionary
        extracted_contours[f"{level} dB Angle"] = angle_column

    # Export data to CSV, saving in the same folder as polars.txt
    csv_file_name = f"contour_lines_data_{os.path.basename(folder_name)}.csv"
    csv_file_path = os.path.join(folder_path, csv_file_name)
    contour_df = pd.DataFrame(extracted_contours)
    contour_df.to_csv(csv_file_path, index=False)
    print(f"Contour data exported to {csv_file_path}")

    return True
