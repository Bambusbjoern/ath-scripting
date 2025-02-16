import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor

# ---------------------------
# USER-CONFIGURABLE VARIABLES
# ---------------------------

# Which regression algorithm to use for fitting the (log10(freq), amplitude) data.
# Options: "linear", "huber", "theilsen"
REGRESSION_ALGORITHM = "huber"

# Which error aggregation metric to use in the rating.
# Options: "rmse", "mae", "median", "huber_loss", "SFM"
ERROR_METRIC = "SFM"

# DELTA is used if "huber_loss" is selected as the aggregator.
DELTA = 1.35

# Frequency ranges for fitting (used to train the line) and for rating.
fit_frequency_range = (3000, 10000)
# Define separate rating sub-ranges, each with its own weight.
rating_ranges = [
    ((3000, 8000), 200.0),   # (frequency range, weight) #100
    ((8001, 12000), 50.0), #50
    ((12001, 20000), 10.0) #10
]

# Global weighting factors for amplitude ratings.
HOR_WEIGHT = 1.0
VER_WEIGHT = 0.8

# Global weighting factor for slope ratings.
GLOBAL_SLOPE_WEIGHT = 0.0

# Define angle-based weightings for amplitude ratings.
angle_weight_hor = {
    '5°': 0.8,
    '10°': 0.8,
    '15°': 0.8,
    '20°': 1.0,
    '25°': 1.0,
    '30°': 1.0,
    '40°': 1.0,
    '50°': 1.0,
    '60°': 1.0
}

angle_weight_ver = {
    '5°': 0.8,
    '10°': 0.8,
    '15°': 0.8,
    '20°': 1.0,
    '25°': 1.0,
    '30°': 1.0,
    '40°': 1.0,
    '50°': 1.0,
    '60°': 1.0
}

# ---------------------------
# SLOPE RATING CONFIGURATION
# ---------------------------
# Target slopes in dB per octave (one octave = frequency doubles).
TARGET_SLOPE_HOR = 0.0   # desired horizontal slope (dB/octave)
TARGET_SLOPE_VER = 0.0   # desired vertical slope (dB/octave)

# Per-angle slope weightings.
slope_weight_hor = {
    '5°': 0.8,
    '10°': 0.8,
    '15°': 0.8,
    '20°': 0.8,
    '25°': 1.0,
    '30°': 1.0,
    '40°': 1.0,
    '50°': 1.0,
    '60°': 1.0,
    '65°': 0.0,
    '70°': 0.0,
    '75°': 0.0,
    '80°': 0.0,
    '85°': 0.0,
    '90°': 0.0,
}

slope_weight_ver = {
    '5°': 0.8,
    '10°': 0.8,
    '15°': 0.8,
    '20°': 0.8,
    '25°': 0.5,
    '30°': 1.0,
    '40°': 1.0,
    '50°': 1.0,
    '60°': 1.0,
    '65°': 0.0,
    '70°': 0.0,
    '75°': 0.0,
    '80°': 0.0,
    '85°': 0.0,
    '90°': 0.0,
}

# --------------------------
# END USER-CONFIGURABLE VARS
# --------------------------

def fit_line_log_scale(freq, amp):
    """
    Fits a (log10(freq), amplitude) line using the selected regression algorithm.
    Returns (slope, intercept).
    """
    log_x = np.log10(freq).reshape(-1, 1)
    if REGRESSION_ALGORITHM.lower() == "linear":
        model = LinearRegression().fit(log_x, amp)
    elif REGRESSION_ALGORITHM.lower() == "theilsen":
        model = TheilSenRegressor().fit(log_x, amp)
    else:
        model = HuberRegressor().fit(log_x, amp)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

def huber_loss(errors, delta=1.0):
    """
    Compute elementwise Huber loss.
    L_delta(a) = 0.5*a^2       if |a| <= delta
                 delta*(|a| - 0.5*delta) otherwise
    """
    abs_err = np.abs(errors)
    quad_part = 0.5 * errors**2
    lin_part = delta * (abs_err - 0.5 * delta)
    return np.where(abs_err <= delta, quad_part, lin_part)

def compute_sfm(amp_db):
    """
    Compute spectral flatness in dB from an array of amplitude in dB.
    - A perfectly flat region → 0 dB.
    - Negative values → less flat (peaks/dips).
    """
    # Convert from dB to linear (assuming amplitude in dB SPL).
    amp_lin = 10**(amp_db / 20.0)
    amp_lin = np.clip(amp_lin, 1e-12, None)

    # Geometric mean of linear amplitudes
    geo_mean = np.exp(np.mean(np.log(amp_lin)))
    # Arithmetic mean
    arith_mean = np.mean(amp_lin)

    if arith_mean <= 1e-12:
        return -999.0  # degenerate case

    sfm = geo_mean / arith_mean
    sfm_db = 10.0 * np.log10(sfm)  # ratio to dB
    return sfm_db

def aggregate_metric(observed, predicted=None):
    """
    Aggregate the "error" based on the selected ERROR_METRIC.
      - For "SFM", we ignore 'predicted' and compute spectral flatness from 'observed' in dB,
        then return -SFM (so higher SFM → smaller 'error').
      - For the other metrics, we compute an error array (observed - predicted) and apply
        RMSE, MAE, Median, or Huber loss, as before.

    Returns a single numeric rating (lower is better).
    """
    if ERROR_METRIC.lower() == "sfm":
        # We interpret 'observed' as dB amplitude; ignoring predicted altogether
        sfm_db = compute_sfm(observed)
        return -sfm_db  # higher SFM yields a lower rating (reward)
    else:
        # "rmse", "mae", "median", "huber_loss", etc.
        errors = observed - predicted
        if ERROR_METRIC.lower() == "rmse":
            mse = np.mean(errors**2)
            return np.sqrt(mse)
        elif ERROR_METRIC.lower() == "mae":
            return np.mean(np.abs(errors))
        elif ERROR_METRIC.lower() == "median":
            return np.median(np.abs(errors))
        elif ERROR_METRIC.lower() == "huber_loss":
            loss_values = huber_loss(errors, delta=DELTA)
            return np.mean(loss_values)
        else:
            # Default to RMSE if user picks unknown metric
            mse = np.mean(errors**2)
            return np.sqrt(mse)

def rate_frequency_response(horns_folder, foldername, simulation_folder, verbose=True):
    """
    Rates the frequency response by processing both horizontal and vertical files.
    For each file, the rating includes:
      1. Amplitude rating (deviation from fitted line or SFM)
      2. Slope rating (difference from target slope)
    Returns a total rating (lower is better) and prints the contributions.
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

    # Accumulators for individual contributions:
    hor_slope_penalty = 0.0
    ver_slope_penalty = 0.0
    hor_amp_rating = 0.0
    ver_amp_rating = 0.0

    def process_file(file, angle_weights, slope_weights, global_dir_weight, is_horizontal=True):
        nonlocal total_rating, hor_slope_penalty, ver_slope_penalty, hor_amp_rating, ver_amp_rating

        basename = os.path.basename(file)
        match = re.search(rf"{foldername}__(?:hor|ver)_deg\+(\d+)\.txt", basename)
        if not match:
            if verbose:
                print(f"Could not extract angle from filename: {file}")
            total_rating += 1000
            return

        angle_label = match.group(1) + "°"
        if angle_label not in angle_weights or angle_label not in slope_weights:
            if verbose:
                print(f"Skipping angle {angle_label}, not found in weighting dictionaries.")
            return

        # Determine slope target and per-angle slope weight.
        if is_horizontal:
            slope_target = TARGET_SLOPE_HOR
            slope_w = slope_weights.get(angle_label, 1.0)
        else:
            slope_target = TARGET_SLOPE_VER
            slope_w = slope_weights.get(angle_label, 1.0)

        final_angle_weight = angle_weights[angle_label] * global_dir_weight

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

        # Create mask for the fit range (used for slope fitting).
        fit_mask = (freq >= fit_frequency_range[0]) & (freq <= fit_frequency_range[1])
        fit_freq = freq[fit_mask]
        fit_amp = amp[fit_mask]
        if fit_freq.size < 2:
            if verbose:
                print(f"Not enough data in fit range for angle {angle_label}.")
            total_rating += 1000
            return

        # Fit the regression line to log10(freq)
        slope_val, intercept = fit_line_log_scale(fit_freq, fit_amp)
        # Compute slope in dB per octave
        slope_db_octave = slope_val * np.log10(2)
        slope_error = abs(slope_db_octave - slope_target)
        slope_penalty = slope_error * slope_w * GLOBAL_SLOPE_WEIGHT
        if verbose:
            print(f"Angle {angle_label}: Slope = {slope_db_octave:.3f} dB/octave, "
                  f"Target = {slope_target}, Slope Penalty = {slope_penalty:.3f}")

        file_rating = slope_penalty  # Start with slope penalty

        # Accumulate slope penalty separately
        if is_horizontal:
            hor_slope_penalty += slope_penalty
        else:
            ver_slope_penalty += slope_penalty

        # Process amplitude or SFM for each frequency sub-range
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

            if ERROR_METRIC.lower() == "sfm":
                # SFM metric: ignore predicted line, compute directly from sub_amp
                sub_err_value = aggregate_metric(observed=sub_amp)
            else:
                # Normal metrics: compare sub_amp to fitted line
                predicted = slope_val * np.log10(sub_freq) + intercept
                sub_err_value = aggregate_metric(observed=sub_amp, predicted=predicted)

            sub_partial = sub_err_value * freq_w * final_angle_weight
            file_rating += sub_partial

        # Accumulate amplitude-related rating (excluding slope penalty)
        if is_horizontal:
            hor_amp_rating += (file_rating - slope_penalty)
        else:
            ver_amp_rating += (file_rating - slope_penalty)

        total_rating += file_rating

    # Process horizontal files
    for file in horizontal_files:
        process_file(file, angle_weight_hor, slope_weight_hor, HOR_WEIGHT, is_horizontal=True)
    # Process vertical files
    for file in vertical_files:
        process_file(file, angle_weight_ver, slope_weight_ver, VER_WEIGHT, is_horizontal=False)

    # Print out detailed contributions
    print("----- Frequency Response Rating Contributions -----")
    print(f"Horizontal Slope Penalty: {hor_slope_penalty:.3f}")
    print(f"Vertical Slope Penalty:   {ver_slope_penalty:.3f}")
    print(f"Horizontal Amplitude Rating: {hor_amp_rating:.3f}")
    print(f"Vertical Amplitude Rating:   {ver_amp_rating:.3f}")
    print("-----------------------------------------------------")

    return round(total_rating, 3)

# Example usage / debugging:
if __name__ == "__main__":
    horns_folder = r"D:\ath\Horns"
    foldername = "2"
    simulation_folder = "ABEC_FreeStanding"
    verbose = True

    rating = rate_frequency_response(horns_folder, foldername, simulation_folder, verbose=verbose)
    print(f"Final rating for simulation {foldername}: {rating}")
