import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Define the frequency ranges
fit_frequency_range = (5000, 8000)
rating_frequency_range = (2000, 20000)

# Define weightings for each angle
weightings_fr = {
    '0°': 1, 
    '5°': 0.0, 
    '20°': 1, 
    '30°': 1, 
    '40°': 1, 
    '50°': 1, 
    '60°': 1
}

# Function to fit a line on a logarithmic frequency scale
def fit_line_log_scale(x, y):
    log_x = np.log10(x)
    model = LinearRegression().fit(log_x.reshape(-1, 1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

# Function to analyze frequency response data
def rate_frequency_response(horns_folder, foldername, simulation_folder, verbose=True):
    file_path = os.path.join(horns_folder, foldername, simulation_folder, "Results", "polars.txt")
    try:
        # Load the data
        polars_df = pd.read_csv(file_path, sep=' ', header=None, names=['Frequency', 'Value', 'Angle_Index'])

        # Convert `Angle_Index` to actual angle labels (assume 19 angles from 0° to 90° evenly spaced)
        angle_step = 90 / 18
        angles = [f"{int(angle):d}°" for angle in np.arange(0, 91, angle_step)]
        polars_df['Angle'] = polars_df['Angle_Index'].apply(lambda x: angles[int(x) - 1])

        # Initialize the total rating accumulator
        total_rating = 0

        for angle, weight in weightings_fr.items():
            # Skip angles with zero weight
            if weight == 0:
                if verbose:
                    print(f"Skipping angle {angle} - weight is 0.")
                continue

            try:
                # Filter data for the current angle
                angle_data = polars_df[polars_df['Angle'] == angle]

                # Extract data for the fit and rating frequency ranges
                fit_data = angle_data[(angle_data['Frequency'] >= fit_frequency_range[0]) & 
                                      (angle_data['Frequency'] <= fit_frequency_range[1])]
                rating_data = angle_data[(angle_data['Frequency'] >= rating_frequency_range[0]) & 
                                         (angle_data['Frequency'] <= rating_frequency_range[1])]

                if len(fit_data) > 1 and len(rating_data) > 1:
                    # Fit the line in the fit frequency range
                    fit_frequencies = fit_data['Frequency'].values
                    fit_values = fit_data['Value'].values
                    slope, intercept = fit_line_log_scale(fit_frequencies, fit_values)

                    # Calculate the deviation in the rating frequency range
                    rating_frequencies = rating_data['Frequency'].values
                    log_rating_frequencies = np.log10(rating_frequencies)
                    rating_values = rating_data['Value'].values
                    deviation = np.abs(rating_values - (slope * log_rating_frequencies + intercept))
                    deviation_rating = np.round(deviation.mean() * weight, 3)

                    # Add the deviation rating to the total rating
                    total_rating += deviation_rating

                    if verbose:
                        print(f"Angle {angle}: Deviation Rating={deviation_rating:.4f}")
                else:
                    if verbose:
                        print(f"Not enough data points for angle {angle} in the fit or rating range.")
                    total_rating += 1000  # Apply penalty if not enough data
            except Exception as e:
                if verbose:
                    print(f"Error processing angle {angle}: {e}")
                total_rating += 1000  # Apply penalty for errors

        return round(total_rating, 3)

    except Exception as e:
        if verbose:
            print(f"Error loading file {file_path}: {e}")
        return 10000  # High penalty for file errors
