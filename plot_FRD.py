import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------
# USER-CONFIGURABLE VARIABLES
# ---------------------------

# If True, the script will perform a quick log-scale fit (like in rate_FRD)
# and plot a dotted line in the same color as the main frequency response.
PLOT_FITTED_LINES = True

# The frequency range to use for fitting the line (e.g., the same as your rate script).
# We'll fit only data in [FIT_LO, FIT_HI], then plot that line across the entire data range.
FIT_LO = 2000
FIT_HI = 14000

# If you prefer ordinary linear regression, set USE_HUBER = False
USE_HUBER = True

# If using Huber, you can set the "epsilon" parameter for HuberRegressor here:
HUBER_EPSILON = 1.35

# You can also set a default y-axis range if you wish
YMIN = -25
YMAX = 5

# --------------------------
# END USER-CONFIGURABLE VARS
# --------------------------

from sklearn.linear_model import LinearRegression, HuberRegressor

def fit_line_log_scale(freq, amp):
    """
    Quick log10-based regression for plotting lines:
      - If USE_HUBER is True, use HuberRegressor(epsilon=HUBER_EPSILON).
      - Else use ordinary LinearRegression.

    Returns slope, intercept
      amplitude ~ slope * log10(freq) + intercept
    """
    log_x = np.log10(freq).reshape(-1, 1)
    if USE_HUBER:
        model = HuberRegressor(epsilon=HUBER_EPSILON).fit(log_x, amp)
    else:
        model = LinearRegression().fit(log_x, amp)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept

def plot_frequency_responses(config_number, sim_folder, horns_folder, results_folder,
                             total_rating, config_id, params_dict):
    """
    Reads frequency response files from the Horns folder and creates a 2×2 figure:

      - Top Left: Horizontal frequency responses, log-scale freq axis, grid lines at 1000Hz increments,
                  optional fitted lines if PLOT_FITTED_LINES=True.
      - Top Right: Vertical frequency responses, same approach.
      - Bottom Left: (Placeholder for DI)
      - Bottom Right: A table of waveguide parameters.

    The figure is then saved in the results folder with filename "{total_rating}_{config_id}.png".
    """
    config_str = str(config_number)  # for filename matching

    # Path to FRD data
    frd_folder = os.path.join(horns_folder, config_str, sim_folder, "Results", "FRD")
    if not os.path.exists(frd_folder):
        print(f"Frequency response folder not found: {frd_folder}")
        return

    # Glob for horizontal and vertical
    hor_pattern = os.path.join(frd_folder, f"{config_str}__hor_deg+*.txt")
    ver_pattern = os.path.join(frd_folder, f"{config_str}__ver_deg+*.txt")
    horizontal_files = sorted(glob.glob(hor_pattern))
    vertical_files = sorted(glob.glob(ver_pattern))

    if not horizontal_files and not vertical_files:
        print("No frequency response files found in the folder.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    def extract_angle(filename, direction):
        # "2__hor_deg+15.txt" => angle="15"
        pattern = rf"{config_str}__{direction}_deg\+(\d+)\.txt"
        match = re.search(pattern, os.path.basename(filename))
        return match.group(1) if match else "?"

    major_ticks = [1000, 2000, 5000, 10000]

    # ---- Horizontal subplot (top-left)
    freq_min_h = float('inf')
    freq_max_h = float('-inf')

    for file in horizontal_files:
        angle = extract_angle(file, "hor")
        data = np.loadtxt(file)
        freq = data[:, 0]
        amp = data[:, 1]

        freq_min_h = min(freq_min_h, freq.min())
        freq_max_h = max(freq_max_h, freq.max())

        # Make 15° and 30° thicker
        lw = 2.5 if angle in ["15", "30", "45", "60"] else 1.0
        # Plot the main response
        line_obj = axs[0, 0].plot(freq, amp, label=f"{angle}°", lw=lw)
        color_used = line_obj[0].get_color()

        # If we want to also plot the fitted line:
        if PLOT_FITTED_LINES:
            # Fit only data in [FIT_LO, FIT_HI]
            mask_fit = (freq >= FIT_LO) & (freq <= FIT_HI)
            freq_fit = freq[mask_fit]
            amp_fit = amp[mask_fit]
            if len(freq_fit) > 1:
                slope, intercept = fit_line_log_scale(freq_fit, amp_fit)
                # We'll plot the line across the entire freq range for visual reference
                sort_idx = np.argsort(freq)
                freq_sorted = freq[sort_idx]
                log_f = np.log10(freq_sorted)
                amp_pred = slope*log_f + intercept
                axs[0, 0].plot(freq_sorted, amp_pred, linestyle=':', lw=1.0, color=color_used)

    axs[0, 0].set_title("Horizontal Frequency Responses")
    axs[0, 0].set_xlabel("Frequency (Hz)")
    axs[0, 0].set_ylabel("Amplitude (dB)")
    axs[0, 0].set_xlim(freq_min_h, freq_max_h)
    axs[0, 0].set_ylim(YMIN, YMAX)
    axs[0, 0].set_xscale('log')
    axs[0, 0].xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    axs[0, 0].xaxis.set_major_formatter(ticker.ScalarFormatter())
    minor_ticks_h = np.arange(np.ceil(freq_min_h/1000)*1000, freq_max_h+1000, 1000)
    axs[0, 0].xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks_h))
    axs[0, 0].xaxis.set_minor_formatter(ticker.NullFormatter())
    axs[0, 0].grid(which='major', axis='x', linestyle='-', color='black', alpha=0.8)
    axs[0, 0].grid(which='minor', axis='x', linestyle=':', color='gray', alpha=0.5)
    axs[0, 0].grid(which='both', axis='y', linestyle='--', color='gray', alpha=0.5)
    axs[0, 0].legend(title="Angle", fontsize=8, title_fontsize=9)

    # ---- Vertical subplot (top-right)
    freq_min_v = float('inf')
    freq_max_v = float('-inf')

    for file in vertical_files:
        angle = extract_angle(file, "ver")
        data = np.loadtxt(file)
        freq = data[:, 0]
        amp = data[:, 1]

        freq_min_v = min(freq_min_v, freq.min())
        freq_max_v = max(freq_max_v, freq.max())

        lw = 2.5 if angle in ["15", "30", "45", "60"] else 1.0
        line_obj = axs[0, 1].plot(freq, amp, label=f"{angle}°", lw=lw)
        color_used = line_obj[0].get_color()

        if PLOT_FITTED_LINES:
            mask_fit = (freq >= FIT_LO) & (freq <= FIT_HI)
            freq_fit = freq[mask_fit]
            amp_fit = amp[mask_fit]
            if len(freq_fit) > 1:
                slope, intercept = fit_line_log_scale(freq_fit, amp_fit)
                sort_idx = np.argsort(freq)
                freq_sorted = freq[sort_idx]
                log_f = np.log10(freq_sorted)
                amp_pred = slope*log_f + intercept
                axs[0, 1].plot(freq_sorted, amp_pred, linestyle=':', lw=1.0, color=color_used)

    axs[0, 1].set_title("Vertical Frequency Responses")
    axs[0, 1].set_xlabel("Frequency (Hz)")
    axs[0, 1].set_ylabel("Amplitude (dB)")
    axs[0, 1].set_xlim(freq_min_v, freq_max_v)
    axs[0, 1].set_ylim(YMIN, YMAX)
    axs[0, 1].set_xscale('log')
    axs[0, 1].xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    axs[0, 1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    minor_ticks_v = np.arange(np.ceil(freq_min_v/1000)*1000, freq_max_v+1000, 1000)
    axs[0, 1].xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks_v))
    axs[0, 1].xaxis.set_minor_formatter(ticker.NullFormatter())
    axs[0, 1].grid(which='major', axis='x', linestyle='-', color='black', alpha=0.8)
    axs[0, 1].grid(which='minor', axis='x', linestyle=':', color='gray', alpha=0.5)
    axs[0, 1].grid(which='both', axis='y', linestyle='--', color='gray', alpha=0.5)
    axs[0, 1].legend(title="Angle", fontsize=8, title_fontsize=9)

    # ---- Bottom Left: Placeholder for DI or other plots
    axs[1, 0].axis('off')
    axs[1, 0].text(0.5, 0.5, "Directivity Index (DI)\n(not implemented)",
                   horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')

    # ---- Bottom Right: Table of waveguide parameters
    axs[1, 1].axis('tight')
    axs[1, 1].axis('off')
    table_data = [[f"{key} = {value}"] for key, value in params_dict.items()]
    param_table = axs[1, 1].table(cellText=table_data,
                                  colLabels=["Parameter"],
                                  cellLoc='left',
                                  loc='center')
    param_table.auto_set_font_size(False)
    param_table.set_fontsize(10)
    axs[1, 1].set_title("Waveguide Parameters", pad=20)

    plt.tight_layout()

    # Save figure
    save_filename = f"{total_rating:.2f}_{config_id}.png"
    save_path = os.path.join(results_folder, save_filename)
    plt.savefig(save_path)
    print(f"Saved image to {save_path}")

    plt.close(fig)


# Example usage:
if __name__ == "__main__":
    config_number = 2
    sim_folder = "ABEC_FreeStanding"
    horns_folder = r"D:\ath\Horns"
    results_folder = r"D:\ath\results"
    total_rating = 123.45
    config_id = 2
    # Some example waveguide parameters
    params_dict = {
        "r0": 14.0,
        "a0": 30.0,
        "a": 60.0,
        "k": 5.0,
        "L": 25.0,
        "s": 1.0,
        "n": 5.0,
        "q": 0.995,
        "va": 20,
        "u_va0": 0.5,
        "u_vk": 0.5,
        "u_vs": 0.5,
        "u_vn": 0.5,
        "mfp": 0.0,
        "mr": 5.0
    }

    plot_frequency_responses(config_number, sim_folder, horns_folder, results_folder,
                             total_rating, config_id, params_dict)
