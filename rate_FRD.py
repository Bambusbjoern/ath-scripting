import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#  Rate Frequency Response from the FRDExport option. Prefixes hor and ver.

# Set file paths
POLARS_FILE = #####

print("how does git work?")

# Define the frequency ranges
fit_frequency_range = (6000, 10000)
rating_frequency_range = (2000, 14000)

# Define weightings for each angle
weightings_fr = {
    '0°': 0.9,
    '5°': 0.0,
    '20°': 0.9,
    '30°': 0.9,
    '40°': 0.9,
    '50°': 0.9,
    '60°': 0.9
}

# Define angles for frequency response data
angle_step = 90 / 18
angles = np.arange(0, 91, angle_step)


# Helper function to fit a line on a logarithmic frequency scale
def fit_line_log_scale(x, y):
    log_x = np.log10(x)
    model = LinearRegression().fit(log_x.reshape(-1, 1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return slope, intercept


# Load data
polars_df = pd.read_csv(POLARS_FILE, sep=' ', header=None, names=['Frequency', 'Value', 'Angle_Index'])
polars_df['Angle'] = polars_df['Angle_Index'].apply(lambda x: f"{angles[int(x) - 1]:.0f}°")

# Pre-filter data for fitting and rating frequency ranges
fit_data = polars_df[(polars_df['Frequency'] >= fit_frequency_range[0]) &
                     (polars_df['Frequency'] <= fit_frequency_range[1])]
rating_data = polars_df[(polars_df['Frequency'] >= rating_frequency_range[0]) &
                        (polars_df['Frequency'] <= rating_frequency_range[1])]


# Function to analyze frequency response data
def analyze_frequency_responses(verbose=True):
    total_rating = 0
    fitted_lines = {}  # Store fitted line data for plotting

    for angle in polars_df['Angle'].unique():
        # Check if the angle has a defined weight in weightings_fr
        if angle not in weightings_fr or weightings_fr[angle] == 0:
            if verbose:
                print(f"Skipping angle {angle} - no weight defined or weight is 0.")
            continue

        try:
            angle_fit_data = fit_data[fit_data['Angle'] == angle]
            angle_rating_data = rating_data[rating_data['Angle'] == angle]

            # Only fit the line if there are enough points
            if len(angle_fit_data) > 1 and len(angle_rating_data) > 1:
                # Fit a straight line on logarithmic x-axis
                slope, intercept = fit_line_log_scale(angle_fit_data['Frequency'].values,
                                                      angle_fit_data['Value'].values)

                # Store fitted line data for plotting
                fitted_lines[angle] = {
                    'slope': slope,
                    'intercept': intercept,
                    'rating_frequencies': angle_rating_data['Frequency'].values,
                    'rating_values': angle_rating_data['Value'].values
                }

                # Calculate deviation for rating
                rating_frequencies = angle_rating_data['Frequency'].values
                log_rating_frequencies = np.log10(rating_frequencies)  # Convert to log scale for rating
                rating_values = angle_rating_data['Value'].values
                fitted_values = slope * log_rating_frequencies + intercept
                deviation = np.abs(rating_values - fitted_values)
                deviation_rating = np.round(deviation.mean() * weightings_fr[angle], 3)
                total_rating += deviation_rating

                if verbose:
                    print(f"Angle: {angle} - Deviation Rating: {deviation_rating:.4f}")
            else:
                if verbose:
                    print(f"Not enough data points for angle {angle} in the fit or rating range.")
                total_rating += 1000  # Apply high penalty if not enough points
        except Exception as e:
            print(f"Error processing angle {angle}: {e}")
            total_rating += 1000  # Apply high penalty in case of error

    return round(total_rating, 3), fitted_lines


# Plot function for frequency response and extended fitted lines
def plot_frequency_responses(fitted_lines):
    plt.figure(figsize=(10, 6))
    for angle, data in fitted_lines.items():
        # Plot actual frequency response data
        plt.plot(data['rating_frequencies'], data['rating_values'], label=f'{angle} Response')

        # Plot fitted line over the rating frequency range
        extended_frequencies = np.linspace(rating_frequency_range[0], rating_frequency_range[1], 500)
        plt.plot(extended_frequencies,
                 data['slope'] * np.log10(extended_frequencies) + data['intercept'],
                 linestyle='--', label=f'{angle} Fitted Line')

    # Highlight fitting and rating frequency ranges
    plt.axvspan(fit_frequency_range[0], fit_frequency_range[1], color='blue', alpha=0.1, label="Fit Range")
    plt.axvspan(rating_frequency_range[0], rating_frequency_range[1], color='orange', alpha=0.1, label="Rating Range")

    plt.xscale('log')
    plt.xlim(1000, 20000)
    plt.ylim(-20, 5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.title("Frequency Response with Fitted Lines (Logarithmic Scale)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


# Execute analysis and plotting
fr_rating, fitted_lines = analyze_frequency_responses()
print(f"\nFrequency Response Rating: {fr_rating}")
plot_frequency_responses(fitted_lines)
