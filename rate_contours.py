import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import configparser

# Load configuration paths from config.ini
config = configparser.ConfigParser()
config.read("config.ini")

HORNS_FOLDER = config["Paths"]["HORNS_FOLDER"]

# Specify frequency ranges for fitting and rating
fit_frequency_range = (3000, 7000)
rating_frequency_range = (2000, 18000)

# Define target slopes and separate weights for deviation and slope for each -xdB line
line_config = {
    "-10 dB Angle": {"weight_deviation": 0.01, "weight_slope": 0.01, "target_slope": 0},
    "-6 dB Angle": {"weight_deviation": 1.0, "weight_slope": 0.3, "target_slope": -0.0},
    "-3 dB Angle": {"weight_deviation": 0.5, "weight_slope": 0.3, "target_slope": -0.0},
    "-2 dB Angle": {"weight_deviation": 0.3, "weight_slope": 0.2, "target_slope": -0.0},
    "-1 dB Angle": {"weight_deviation": 0.2, "weight_slope": 0.2, "target_slope": 0.0}
}

# Define the target value for "-6 dB Angle" in the fit frequency range
target_6dB_angle = 50
weight_6dB_fit_deviation = 0.01

# Function to fit a line (linear regression) to x and y data
def fit_line(x, y):
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    return model.coef_[0], model.intercept_  # Return slope and intercept

# Rating function to evaluate each CSV file
def rate_contours(foldername):
    horn_path = os.path.join(HORNS_FOLDER, foldername, "ABEC_FreeStanding", "Results")
    
    # Check if Results folder exists
    if not os.path.isdir(horn_path):
        return None
    
    # Look for contour_lines_data CSV files
    for file in os.listdir(horn_path):
        if file.startswith("contour_lines_data") and file.endswith(".csv"):
            file_path = os.path.join(horn_path, file)
            data = pd.read_csv(file_path)

            # Fit a line for each -xdB line in the fit frequency range
            fit_data = data[(data["Frequency"] >= fit_frequency_range[0]) & (data["Frequency"] <= fit_frequency_range[1])]
            fit_lines = {}
            slopes = {}
            
            for line, config in line_config.items():
                if line in fit_data.columns:
                    frequencies = fit_data["Frequency"].values
                    values = fit_data[line].values
                    slope, intercept = fit_line(frequencies, values)
                    fit_lines[line] = (slope, intercept)  # Store slope and intercept
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

            # Sum up all weighted deviations and slope penalties
            total_rating = sum(r for r in ratings.values() if not np.isnan(r)) + sum(slope_penalties.values())
            
            # Return the total rating for this file
            return total_rating

    # If no valid file was processed, return None
    return None

if __name__ == "__main__":
    # Iterate over all folders in HORNS_FOLDER and calculate ratings
    ratings = {}
    for foldername in os.listdir(HORNS_FOLDER):
        total_rating = rate_simulation(foldername)
        if total_rating is not None:
            ratings[foldername] = total_rating
            print(f"Folder: {foldername}, Total Rating: {total_rating:.2f}")
        else:
            print(f"Folder: {foldername} has no valid simulation data.")

    # Print a summary of all ratings
    print("\nSummary of Ratings:")
    for folder, rating in ratings.items():
        print(f"  {folder}: {rating:.2f}")
