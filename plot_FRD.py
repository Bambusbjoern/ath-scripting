import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re

def plot_frequency_responses(config_number, sim_folder, horns_folder, results_folder, total_rating, config_id, params_dict):
    """
    Reads frequency response files from the Horns folder and creates a 2×2 figure:

      - Top Left: Horizontal frequency responses (with x-axis logarithmic, major ticks labeled at 1000, 2000, 5000, and 10000 Hz,
                  with gridlines every 1000Hz; gridlines corresponding to these major ticks are solid while others are dotted),
                  and y-axis from -25dB to 5dB).
      - Top Right: Vertical frequency responses (with the same x-axis settings as above).
      - Bottom Left: (DI vs. Angle plot - currently non-functional and commented out).
      - Bottom Right: A table listing the waveguide parameters (one per line).

    The figure is then saved in the results folder with a filename like {total_rating}_{config_id}.png.

    Parameters:
      config_number (int or str): The configuration number (e.g., 2).
      sim_folder (str): The simulation folder name (e.g., "ABEC_FreeStanding").
      horns_folder (str): The base folder for Horns data (e.g., "D:\\ath\\Horns").
      results_folder (str): Folder where the image should be saved (from config.ini).
      total_rating (float): The overall rating, used in the filename.
      config_id (int or str): The configuration ID, used in the filename.
      params_dict (dict): A dictionary with the waveguide parameters for the simulation.
    """

    # Ensure config_number is a string for filename patterns.
    config_str = str(config_number)

    # Build the path to the frequency response data.
    frd_folder = os.path.join(horns_folder, config_str, sim_folder, "Results", "FRD")
    if not os.path.exists(frd_folder):
        print(f"Frequency response folder not found: {frd_folder}")
        return

    # Use glob to locate horizontal and vertical response files.
    hor_pattern = os.path.join(frd_folder, f"{config_str}__hor_deg+*.txt")
    ver_pattern = os.path.join(frd_folder, f"{config_str}__ver_deg+*.txt")
    horizontal_files = sorted(glob.glob(hor_pattern))
    vertical_files = sorted(glob.glob(ver_pattern))

    if not horizontal_files and not vertical_files:
        print("No frequency response files found in the folder.")
        return

    # Create a 2×2 grid of subplots.
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Helper function to extract the angle from a filename.
    def extract_angle(filename, direction):
        # Expected filename format: "2__hor_deg+15.txt" or "2__ver_deg+10.txt"
        pattern = rf"{config_str}__{direction}_deg\+(\d+)\.txt"
        match = re.search(pattern, os.path.basename(filename))
        return match.group(1) if match else "?"

    # Define the major ticks that will be labeled.
    major_ticks = [1000, 2000, 5000, 10000]

    # ---- Top Left: Horizontal Frequency Responses ----
    freq_min_h = float('inf')
    freq_max_h = float('-inf')
    for file in horizontal_files:
        angle = extract_angle(file, "hor")
        data = np.loadtxt(file)
        frequency = data[:, 0]
        amplitude = data[:, 1]
        # Update the global frequency range.
        freq_min_h = min(freq_min_h, frequency.min())
        freq_max_h = max(freq_max_h, frequency.max())
        # Use a thicker line for 15° and 30°.
        lw = 2.5 if angle in ["15", "30"] else 1.0
        axs[0, 0].plot(frequency, amplitude, label=f"{angle}°", lw=lw)
    axs[0, 0].set_title("Horizontal Frequency Responses")
    axs[0, 0].set_xlabel("Frequency (Hz)")
    axs[0, 0].set_ylabel("Amplitude (dB)")
    axs[0, 0].set_xlim(freq_min_h, freq_max_h)
    axs[0, 0].set_ylim(-25, 5)
    axs[0, 0].set_xscale('log')
    # Set major ticks only at specified frequencies.
    axs[0, 0].xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    axs[0, 0].xaxis.set_major_formatter(ticker.ScalarFormatter())
    # Set minor ticks every 1000 Hz over the full range.
    minor_ticks_h = np.arange(np.ceil(freq_min_h/1000)*1000, freq_max_h+1000, 1000)
    axs[0, 0].xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks_h))
    # Disable labels on minor ticks.
    axs[0, 0].xaxis.set_minor_formatter(ticker.NullFormatter())
    # Draw grid: solid for major ticks, dotted for minor ticks.
    axs[0, 0].grid(which='major', axis='x', linestyle='-', color='black', alpha=0.8)
    axs[0, 0].grid(which='minor', axis='x', linestyle=':', color='gray', alpha=0.5)
    axs[0, 0].grid(which='both', axis='y', linestyle='--', color='gray', alpha=0.5)
    axs[0, 0].legend(title="Angle", fontsize=8, title_fontsize=9)

    # ---- Top Right: Vertical Frequency Responses ----
    freq_min_v = float('inf')
    freq_max_v = float('-inf')
    for file in vertical_files:
        angle = extract_angle(file, "ver")
        data = np.loadtxt(file)
        frequency = data[:, 0]
        amplitude = data[:, 1]
        freq_min_v = min(freq_min_v, frequency.min())
        freq_max_v = max(freq_max_v, frequency.max())
        lw = 2.5 if angle in ["15", "30"] else 1.0
        axs[0, 1].plot(frequency, amplitude, label=f"{angle}°", lw=lw)
    axs[0, 1].set_title("Vertical Frequency Responses")
    axs[0, 1].set_xlabel("Frequency (Hz)")
    axs[0, 1].set_ylabel("Amplitude (dB)")
    axs[0, 1].set_xlim(freq_min_v, freq_max_v)
    axs[0, 1].set_ylim(-25, 5)
    axs[0, 1].set_xscale('log')
    # Set major ticks.
    axs[0, 1].xaxis.set_major_locator(ticker.FixedLocator(major_ticks))
    axs[0, 1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    # Set minor ticks every 1000 Hz.
    minor_ticks_v = np.arange(np.ceil(freq_min_v/1000)*1000, freq_max_v+1000, 1000)
    axs[0, 1].xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks_v))
    axs[0, 1].xaxis.set_minor_formatter(ticker.NullFormatter())
    # Draw grid: solid for major ticks, dotted for minor ticks.
    axs[0, 1].grid(which='major', axis='x', linestyle='-', color='black', alpha=0.8)
    axs[0, 1].grid(which='minor', axis='x', linestyle=':', color='gray', alpha=0.5)
    axs[0, 1].grid(which='both', axis='y', linestyle='--', color='gray', alpha=0.5)
    axs[0, 1].legend(title="Angle", fontsize=8, title_fontsize=9)

    # ---- Bottom Left: Directivity Index (DI) vs. Angle ----
    # (This section is currently non-functional and has been commented out.)
    """
    di_angles = []
    di_values = []
    for file in horizontal_files:
        angle = extract_angle(file, "hor")
        data = np.loadtxt(file)
        di = np.mean(data[:, 1])  # Dummy calculation—replace with your DI calculation.
        di_angles.append(float(angle))
        di_values.append(di)
    di_angles, di_values = zip(*sorted(zip(di_angles, di_values)))
    axs[1, 0].plot(di_angles, di_values, marker='o', linestyle='-', color='purple')
    axs[1, 0].set_title("Directivity Index (DI) vs. Angle")
    axs[1, 0].set_xlabel("Angle (°)")
    axs[1, 0].set_ylabel("DI (Avg. Amplitude)")
    axs[1, 0].grid(True)
    """
    # Instead, leave a placeholder text in the bottom left subplot.
    axs[1, 0].axis('off')
    axs[1, 0].text(0.5, 0.5, "Directivity Index (DI) plot\n(not implemented)",
                   horizontalalignment='center', verticalalignment='center', fontsize=12, color='gray')

    # ---- Bottom Right: Waveguide Parameters Table ----
    axs[1, 1].axis('tight')
    axs[1, 1].axis('off')
    # Create table data with one parameter per row.
    table_data = [[f"{key} = {value}"] for key, value in params_dict.items()]
    table = axs[1, 1].table(cellText=table_data,
                            colLabels=["Parameter"],
                            cellLoc='left',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    axs[1, 1].set_title("Waveguide Parameters", pad=20)

    # Adjust layout to prevent overlap.
    plt.tight_layout()

    # Save the figure in the results folder with filename "{total_rating}_{config_id}.png".
    save_filename = f"{total_rating:.2f}_{config_id}.png"
    save_path = os.path.join(results_folder, save_filename)
    plt.savefig(save_path)
    print(f"Saved image to {save_path}")

    # Close the figure to avoid displaying it.
    plt.close(fig)


# Example usage:
if __name__ == "__main__":
    # These example values would typically be provided by your main workflow.
    config_number = 2
    sim_folder = "ABEC_FreeStanding"
    horns_folder = r"D:\ath\Horns"
    results_folder = r"D:\ath\results"  # As specified in config.ini
    total_rating = 123.45  # Example overall rating
    config_id = 2  # Example configuration ID

    # Example waveguide parameters (replace with your actual parameters)
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

    plot_frequency_responses(config_number, sim_folder, horns_folder, results_folder, total_rating, config_id, params_dict)
