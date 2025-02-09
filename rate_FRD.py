import os
import glob
import re
import numpy as np
import pandas as pd

# ---------------------------
# USER-CONFIGURABLE VARIABLES
# ---------------------------

# Which regression algorithm to use for fitting the (log10(freq), amplitude) data.
# Possible options:
#   - "linear"    => Ordinary Least Squares
#   - "huber"     => HuberRegressor (robust to outliers)
#   - "theilsen"  => TheilSenRegressor (robust to outliers)
REGRESSION_ALGORITHM = "linear"

# Which error aggregation metric to use in the rating.
# Possible options:
#   - "rmse"       => Root Mean Squared Error
#   - "mae"        => Mean Absolute Error
#   - "median"     => Median Absolute Error
#   - "huber_loss" => Huber loss aggregator
ERROR_METRIC = "median"

# DELTA is used in multiple contexts if "huber" approaches are chosen:
#   1) If you're using "huber" as the regression algorithm (HuberRegressor),
#      you could pass 'epsilon=DELTA' to the constructor if you want (by default it's 1.35).
#   2) If you're using "huber_loss" as the aggregator (ERROR_METRIC), DELTA determines the
#      threshold between quadratic and linear penalty for outliers in the aggregator.
#         - A smaller DELTA means you treat more values as outliers (less quadratic region).
#         - A larger DELTA means you treat fewer values as outliers (larger quadratic region).
DELTA = 1.35

# Frequency ranges for fitting (used to train the line) and for rating (used to evaluate the deviation).
fit_frequency_range = (2000, 10000)
# You can define separate rating sub-ranges, each with its own weighting.
rating_ranges = [
    ((2000, 8000), 1.5),   # (freq_range, weight)
    ((8001, 12000), 0.5),
    ((12001, 20000), 0.1)
]

# Global weighting factors for horizontal and vertical responses:
HOR_WEIGHT = 1.0
VER_WEIGHT = 0.5

# Define angle-based weightings for horizontal angles.
angle_weight_hor = {
    '5°': 1.0,
    '10°': 1.0,
    '15°': 1.0,
    '20°': 1.0,
    '25°': 1.0,
    '30°': 1.0,
    '40°': 0.5,
    '50°': 0.5,
    '60°': 0.5
}

# Define angle-based weightings for vertical angles.
angle_weight_ver = {
    '5°': 1.0,
    '10°': 1.0,
    '20°': 1.0,
    '30°': 1.0,
    '40°': 1.0,
    '50°': 1.0,
    '60°': 1.0,
}

# --------------------------
# END USER-CONFIGURABLE VARS
# --------------------------

from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor

def fit_line_log_scale(freq, amp):
    """
    Fits a (log10(freq), amplitude) line using the user-selected algorithm.
    Returns (slope, intercept).
    """
    log_x = np.log10(freq).reshape(-1, 1)

    if REGRESSION_ALGORITHM.lower() == "linear":
        model = LinearRegression().fit(log_x, amp)
    elif REGRESSION_ALGORITHM.lower() == "theilsen":
        model = TheilSenRegressor().fit(log_x, amp)
    else:
        # Default to "huber"
        model = HuberRegressor().fit(log_x, amp)

    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

def compute_errors(predicted, observed):
    """
    Returns an array of errors: observed - predicted.
    """
    return observed - predicted

def huber_loss(errors, delta=1.0):
    """
    Compute elementwise Huber loss for an array of errors.
    L_delta(a) = 0.5*a^2       if |a| <= delta
                 delta*(|a| - 0.5*delta) otherwise

    The parameter 'delta' controls the threshold at which we switch
    from quadratic to linear penalty for large errors.
    """
    abs_err = np.abs(errors)
    quad_part = 0.5 * errors**2
    lin_part = delta * (abs_err - 0.5*delta)
    return np.where(abs_err <= delta, quad_part, lin_part)

def aggregate_errors(errors):
    """
    Applies the user-selected ERROR_METRIC to the given array of errors
    and returns a single float rating for those errors.
    """
    if ERROR_METRIC.lower() == "rmse":
        mse = np.mean(errors**2)
        return np.sqrt(mse)
    elif ERROR_METRIC.lower() == "mae":
        return np.mean(np.abs(errors))
    elif ERROR_METRIC.lower() == "median":
        return np.median(np.abs(errors))
    elif ERROR_METRIC.lower() == "huber_loss":
        # Use our custom huber_loss function:
        loss_values = huber_loss(errors, delta=DELTA)
        return np.mean(loss_values)
    else:
        # Default to RMSE
        mse = np.mean(errors**2)
        return np.sqrt(mse)

def rate_frequency_response(horns_folder, foldername, simulation_folder, verbose=True):
    """
    Rates the frequency response. For each horizontal or vertical file:
      1) Identify angle and direction (hor_deg+X or ver_deg+X).
      2) Use angle_weight_hor or angle_weight_ver for angle weighting.
      3) Multiply by HOR_WEIGHT or VER_WEIGHT, respectively.
      4) For each sub-range in rating_ranges, compute predicted vs. observed,
         gather errors, apply aggregator, multiply by sub-range's weighting,
         sum => partial rating for that file.
      5) Summation of partial ratings over all sub-ranges => final file rating.
      6) Sum across all files => total rating.

    A high penalty (1000) is added if not enough data is present or if any error occurs.

    Return:
      total_rating (float), where lower is better.
    """
    frd_path = os.path.join(horns_folder, foldername, simulation_folder, "Results", "FRD")
    if not os.path.exists(frd_path):
        if verbose:
            print(f"FRD folder not found: {frd_path}")
        return 10000

    hor_pattern = os.path.join(frd_path, f"{foldername}__hor_deg+*.txt")
    ver_pattern = os.path.join(frd_path, f"{foldername}__ver_deg+*.txt")
    horizontal_files = sorted(glob.glob(hor_pattern))
    vertical_files = sorted(glob.glob(ver_pattern))

    if not horizontal_files and not vertical_files:
        if verbose:
            print("No frequency response files found.")
        return 10000

    total_rating = 0.0

    def process_file(file, angle_weights, global_dir_weight):
        nonlocal total_rating

        # Extract angle from filename
        match = re.search(rf"{foldername}__(?:hor|ver)_deg\+(\d+)\.txt", os.path.basename(file))
        if not match:
            if verbose:
                print(f"Could not extract angle from filename: {file}")
            total_rating += 1000
            return

        angle_label = match.group(1) + "°"
        if angle_label not in angle_weights:
            if verbose:
                print(f"Skipping angle {angle_label}, not in angle_weights.")
            return

        # Combine angle weighting with the global weighting for direction
        final_angle_weight = angle_weights[angle_label] * global_dir_weight

        # Load the data
        try:
            data = np.loadtxt(file)
        except Exception as e:
            if verbose:
                print(f"Error reading {file}: {e}")
            total_rating += 1000
            return

        if data.ndim == 1 or data.shape[0] < 2:
            if verbose:
                print(f"Not enough data in file for angle {angle_label}.")
            total_rating += 1000
            return

        freq = data[:, 0]
        amp = data[:, 1]

        # Fit range
        fit_mask = (freq >= fit_frequency_range[0]) & (freq <= fit_frequency_range[1])
        fit_freq = freq[fit_mask]
        fit_amp = amp[fit_mask]

        if fit_freq.size < 2:
            if verbose:
                print(f"Not enough data points in fit range for angle {angle_label}.")
            total_rating += 1000
            return

        # Fit the line on log10 scale
        slope, intercept = fit_line_log_scale(fit_freq, fit_amp)

        # Now compute partial rating for each sub-range
        file_rating = 0.0
        for (freq_range, freq_w) in rating_ranges:
            lo, hi = freq_range
            mask = (freq >= lo) & (freq <= hi)
            sub_freq = freq[mask]
            sub_amp = amp[mask]
            if sub_freq.size < 2:
                if verbose:
                    print(f"No data in rating range {freq_range} for angle {angle_label}.")
                file_rating += 1000
                continue

            predicted = slope * np.log10(sub_freq) + intercept
            errs = sub_amp - predicted
            # Use the aggregator:
            sub_err_value = aggregate_errors(errs)
            # Weighted by freq_w and final_angle_weight
            partial = sub_err_value * freq_w * final_angle_weight
            file_rating += partial

        total_rating += file_rating

    # Process horizontal files
    for file in horizontal_files:
        process_file(file, angle_weight_hor, HOR_WEIGHT)

    # Process vertical files
    for file in vertical_files:
        process_file(file, angle_weight_ver, VER_WEIGHT)

    return round(total_rating, 3)

# Example usage:
if __name__ == "__main__":
    horns_folder = r"D:\ath\Horns"
    foldername = "2"
    simulation_folder = "ABEC_FreeStanding"
    verbose = True

    rating = rate_frequency_response(horns_folder, foldername, simulation_folder, verbose=verbose)
    print(f"Final rating for simulation {foldername}: {rating}")
