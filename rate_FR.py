import os
import numpy as np
import glob
import re
import pandas as pd
from sklearn.linear_model import LinearRegression

# Define the frequency ranges for fitting and rating
fit_frequency_range = (5000, 8000)  # Frequencies (Hz) used to perform the linear fit
rating_frequency_range = (2000, 20000)  # Frequencies (Hz) used to evaluate the deviation

# Define weightings for selected angles that should be rated.
# Only these angles will be processed.
weightings_fr = {
    '0°': 1,
    '5°': 1,
    '20°': 1,
    '30°': 1,
    '40°': 1,
    '50°': 1,
    '60°': 1
}


def fit_line_log_scale(x, y):
    """
    Fits a linear model to the data on a logarithmic frequency scale.
    Returns the slope and intercept of the fitted line.

    Parameters:
      x : array-like, frequencies.
      y : array-like, amplitude values.
    """
    log_x = np.log10(x)
    model = LinearRegression().fit(log_x.reshape(-1, 1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept


def rate_frequency_response(horns_folder, foldername, simulation_folder, verbose=True):
    """
    Rates the frequency response based on horizontal response files in the FRD folder.

    This function expects to find horizontal files in:
      horns_folder / foldername / simulation_folder / "Results" / "FRD"
    Files should follow the naming scheme: "{foldername}__hor_deg+<angle>.txt"
    Each file must contain three columns: frequency, amplitude, and phase.

    For each file:
      - The angle is extracted from the filename.
      - Only files corresponding to angles specified in weightings_fr are processed.
      - Data is filtered to obtain two subsets:
          * The fit data (for frequencies between fit_frequency_range[0] and fit_frequency_range[1]).
          * The rating data (for frequencies between rating_frequency_range[0] and rating_frequency_range[1]).
      - A linear regression is performed on log10(frequency) vs. amplitude for the fit data.
      - The average absolute deviation (in the rating range) is computed and weighted.

    A high penalty (1000) is added if not enough data is present or if an error occurs.

    Returns:
      The total rating (a lower rating is better).
    """
    # Construct the path to the FRD folder.
    frd_path = os.path.join(horns_folder, foldername, simulation_folder, "Results", "FRD")
    if not os.path.exists(frd_path):
        if verbose:
            print(f"FRD folder not found: {frd_path}")
        return 10000  # High penalty if folder is missing

    # Find all horizontal response files using glob.
    hor_pattern = os.path.join(frd_path, f"{foldername}__hor_deg+*.txt")
    horizontal_files = sorted(glob.glob(hor_pattern))

    if len(horizontal_files) == 0:
        if verbose:
            print("No horizontal frequency response files found.")
        return 10000  # High penalty if no files found

    total_rating = 0.0

    # Process each horizontal file.
    for file in horizontal_files:
        # Extract the angle from the filename.
        # Expected filename example: "2__hor_deg+15.txt"
        match = re.search(rf"{foldername}__hor_deg\+(\d+)\.txt", os.path.basename(file))
        if match:
            angle_label = match.group(1) + "°"
        else:
            if verbose:
                print(f"Could not extract angle from filename: {file}")
            continue

        # Only rate files with angles specified in weightings_fr.
        if angle_label not in weightings_fr:
            if verbose:
                print(f"Skipping angle {angle_label} as it is not in the rating list.")
            continue

        # Get the weighting for this angle.
        weight = weightings_fr[angle_label]

        try:
            # Load the data (assumes whitespace-separated values)
            data = np.loadtxt(file)
            # If data has only one row, skip it with a penalty.
            if data.ndim == 1 or data.shape[0] < 2:
                if verbose:
                    print(f"Not enough data in file for angle {angle_label}.")
                total_rating += 1000
                continue

            frequencies = data[:, 0]
            amplitudes = data[:, 1]

            # Filter data for the fit frequency range.
            fit_mask = (frequencies >= fit_frequency_range[0]) & (frequencies <= fit_frequency_range[1])
            rating_mask = (frequencies >= rating_frequency_range[0]) & (frequencies <= rating_frequency_range[1])
            fit_frequencies = frequencies[fit_mask]
            fit_amplitudes = amplitudes[fit_mask]
            rating_frequencies = frequencies[rating_mask]
            rating_amplitudes = amplitudes[rating_mask]

            if len(fit_frequencies) > 1 and len(rating_frequencies) > 1:
                # Perform the linear regression on the log10 scale.
                slope, intercept = fit_line_log_scale(fit_frequencies, fit_amplitudes)
                # Predict the amplitudes in the rating frequency range.
                log_rating_frequencies = np.log10(rating_frequencies)
                predicted = slope * log_rating_frequencies + intercept
                # Compute the average absolute deviation.
                deviation = np.abs(rating_amplitudes - predicted)
                deviation_rating = np.round(deviation.mean() * weight, 3)
                total_rating += deviation_rating
                if verbose:
                    print(f"Angle {angle_label}: Deviation Rating = {deviation_rating:.4f}")
            else:
                if verbose:
                    print(f"Not enough data points for angle {angle_label} in fit or rating range.")
                total_rating += 1000  # Penalty if insufficient data
        except Exception as e:
            if verbose:
                print(f"Error processing angle {angle_label}: {e}")
            total_rating += 1000  # Penalty for errors

    return round(total_rating, 3)


# Example usage:
if __name__ == "__main__":
    # These example values would typically be provided by your main workflow.
    horns_folder = r"D:\ath\Horns"  # Base folder for Horns data
    foldername = "2"  # Configuration number as a string (used in file naming)
    simulation_folder = "ABEC_FreeStanding"  # Simulation folder type
    verbose = True

    total_rating = rate_frequency_response(horns_folder, foldername, simulation_folder, verbose)
    print(f"Total Frequency Response Rating: {total_rating}")
