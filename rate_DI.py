import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Define the frequency ranges
fit_frequency_range = (5000, 8000)
rating_frequency_range = (3000, 16000)

# Define weightings and target slopes for each DI line
target_slopes_di = {
    'DI_0': 0.25,
    'DI_10': 0.25, 
    'DI_20': 0.25
}  # Slopes in dB per octave
weightings_di_deviation = {
    'DI_0': 0.3, 
    'DI_10': 3.0, 
    'DI_20': 0.5
}
weightings_di_slope = {
    'DI_0': 0.7, 
    'DI_10': 2.5, 
    'DI_20': 0.4
}

# Function to fit a line on a logarithmic frequency scale in dB/octave
def fit_line_db_octave(x, y):
    log_x = np.log2(x)
    model = LinearRegression().fit(log_x.reshape(-1, 1), y)
    slope_db_per_octave = model.coef_[0]
    intercept = model.intercept_
    return slope_db_per_octave, intercept

# Function to analyze DI data
def rate_DI(horns_folder, foldername, simulation_folder, verbose=True):
    file_path = os.path.join(horns_folder, foldername, simulation_folder, "Results", "DI.txt")
    try:
        # Load DI data
        di_df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['Frequency', 'DI_0', 'DI_10', 'DI_20'])
        total_di_rating = 0

        for di_column in ['DI_0', 'DI_10', 'DI_20']:
            try:
                # Extract data for the fit and rating frequency ranges
                fit_data = di_df[(di_df['Frequency'] >= fit_frequency_range[0]) & (di_df['Frequency'] <= fit_frequency_range[1])]
                rating_data = di_df[(di_df['Frequency'] >= rating_frequency_range[0]) & (di_df['Frequency'] <= rating_frequency_range[1])]

                # Calculate slope and intercept using fit range
                fit_frequencies = fit_data['Frequency'].values
                fit_values = fit_data[di_column].values
                slope_db_octave, intercept = fit_line_db_octave(fit_frequencies, fit_values)

                # Calculate deviation from the fitted line in the rating range
                rating_frequencies = rating_data['Frequency'].values
                log_rating_frequencies = np.log2(rating_frequencies)
                rating_values = rating_data[di_column].values
                deviation = np.abs(rating_values - (slope_db_octave * log_rating_frequencies + intercept))
                deviation_rating = np.round(deviation.mean() * weightings_di_deviation[di_column], 3)

                # Calculate slope rating based on target slope in dB/octave
                slope_rating = np.round(np.abs(slope_db_octave - target_slopes_di[di_column]) * weightings_di_slope[di_column], 3)

                total_di_rating += deviation_rating + slope_rating
                if verbose:
                    print(f"{di_column}: Slope={slope_db_octave:.4f} dB/octave, Deviation Rating={deviation_rating:.4f}, Slope Rating={slope_rating:.4f}")

            except Exception as e:
                if verbose:
                    print(f"Error processing {di_column}: {e}")
                total_di_rating += 1000  # Apply high penalty in case of error

        return round(total_di_rating, 3)

    except Exception as e:
        if verbose:
            print(f"Error loading file {file_path}: {e}")
        return 10000  # Return a very high penalty if file loading fails
