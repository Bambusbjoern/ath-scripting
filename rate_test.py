import os
import shutil
import pandas as pd
import numpy as np
import configparser
from sklearn.linear_model import LinearRegression

# Load configuration paths from config.ini
config = configparser.ConfigParser()
config.read("config.ini")

HORNS_FOLDER = config["Paths"]["HORNS_FOLDER"]
RESULTS_FOLDER = config["Paths"]["RESULTS_FOLDER"]

# Specify frequency ranges for fitting and rating
fit_frequency_range = (3000, 7000)
rating_frequency_range = (3000, 12000)

# Define target slopes and separate weights for deviation and slope for each -xdB line
line_config = {
    "-10 dB Angle": {"weight_deviation": 0.01, "weight_slope": 0.01, "target_slope": 0},
    "-6 dB Angle": {"weight_deviation": 1.0, "weight_slope": 0.1, "target_slope": -0.0},
    "-3 dB Angle": {"weight_deviation": 0.5, "weight_slope": 0.1, "target_slope": -0.0},
    "-2 dB Angle": {"weight_deviation": 0.3, "weight_slope": 0.1, "target_slope": -0.0},
    "-1 dB Angle": {"weight_deviation": 0.2, "weight_slope": 0.1, "target_slope": 0.0}
}

# Define a penalty value for lines that cannot be fitted due to insufficient data points
fitting_penalty = 100  # Adjust this value based on desired penalty severity

# Define the target value for "-6 dB Angle" in the fit frequency range
target_6dB_angle = 50
weight_6dB_fit_deviation = 0.01

# Function to fit a line (linear regression) to x and y data, applying a fitting penalty if fitting fails
def fit_line(x, y):
    """Fit a line (linear regression) to the provided x and y values, handling NaNs and applying penalty if fitting fails."""
    mask = ~np.isnan(y)  # Create a mask where y is not NaN
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    if len(x_filtered) < 2:
        # Apply penalty if there aren't enough points after filtering
        print("Not enough valid data points for fitting. Applying fitting penalty.")
        return None, fitting_penalty
    
    model = LinearRegression()
    model.fit(x_filtered.reshape(-1, 1), y_filtered)
    return model.coef_[0], model.intercept_  # Return slope and intercept

# Rating function to evaluate each CSV file
def rate_previous_simulations():
    for foldername in os.listdir(HORNS_FOLDER):
        horn_path = os.path.join(HORNS_FOLDER, foldername, "ABEC_FreeStanding", "Results")
        
        if not os.path.isdir(horn_path):
            continue
        
        for file in os.listdir(horn_path):
            if file.startswith("contour_lines_data") and file.endswith(".csv"):
                file_path = os.path.join(horn_path, file)
                data = pd.read_csv(file_path)

                # Fit a line for each -xdB line in the fit frequency range
                fit_data = data[(data["Frequency"] >= fit_frequency_range[0]) & (data["Frequency"] <= fit_frequency_range[1])]
                fit_lines = {}
                slopes = {}
                penalties = 0
                
                for line, config in line_config.items():
                    if line in fit_data.columns:
                        frequencies = fit_data["Frequency"].values
                        values = fit_data[line].values
                        slope, intercept = fit_line(frequencies, values)
                        
                        if slope is None:
                            # Add fitting penalty if line fitting failed
                            penalties += fitting_penalty
                        else:
                            fit_lines[line] = (slope, intercept)
                            slopes[line] = slope

                # Calculate slope deviation penalty for each line
                slope_penalties = {}
                for line, config in line_config.items():
                    if line in slopes:
                        slope_deviation = abs(slopes[line] - config["target_slope"])
                        slope_penalty = slope_deviation * config["weight_slope"]
                        slope_penalties[line] = slope_penalty
                    else:
                        slope_penalties[line] = 0

                # Calculate deviations from the fitted line in the rating frequency range
                rate_data = data[(data["Frequency"] >= rating_frequency_range[0]) & (data["Frequency"] <= rating_frequency_range[1])]
                ratings = {}
                for line, config in line_config.items():
                    if line in rate_data.columns and line in fit_lines:
                        frequencies = rate_data["Frequency"].values
                        values = rate_data[line].values
                        slope, intercept = fit_lines[line]
                        fitted_values = slope * frequencies + intercept
                        deviation = np.abs(values - fitted_values)
                        mean_deviation = deviation.mean()
                        weighted_rating = mean_deviation * config["weight_deviation"]
                        ratings[line] = weighted_rating
                    else:
                        ratings[line] = np.nan

                # Sum up all weighted deviations, slope penalties, and fitting penalties
                total_rating = sum(r for r in ratings.values() if not np.isnan(r)) + sum(slope_penalties.values()) + penalties

                # Check if the total rating is 0.00 and apply a high penalty if so
                if np.isnan(total_rating) or total_rating == 0.0:
                    print(f"Detected 0.00 rating for {file_path}. Applying zero-rating penalty.")
                    total_rating = 1000  # Apply a high penalty if the total rating is 0.00
                
                # Construct output filename with rating prefix
                rating_str = f"{total_rating:.2f}"
                output_filename = f"{rating_str}_{file}"
                output_path = os.path.join(RESULTS_FOLDER, output_filename)
                
                # Copy associated PNG file with the rating prefix to RESULTS_FOLDER
                png_file_name = f"{foldername}.png"
                png_file_path = os.path.join(horn_path, png_file_name)
                if os.path.isfile(png_file_path):
                    new_png_file_name = f"{rating_str}_{foldername}.png"
                    new_png_file_path = os.path.join(RESULTS_FOLDER, new_png_file_name)
                    shutil.copy(png_file_path, new_png_file_path)
                    print(f"Copied PNG file to: {new_png_file_path}")
                else:
                    print(f"No PNG file found for folder: {foldername}")

                # Display the total rating for the file
                print(f"File: {file_path}, Rating: {total_rating:.2f}")

if __name__ == "__main__":
    # Ensure the results folder exists
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Rate previous simulations and copy associated files
    rate_previous_simulations()
