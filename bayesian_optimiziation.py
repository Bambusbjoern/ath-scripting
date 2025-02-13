import os
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
import configparser
import re

from generate_config import generate_waveguide_config
from generate_abec_file import generate_abec_file
from run_abec import run_abec_simulation
from generate_report import generate_report
from rate_FRD import rate_frequency_response
from get_simulation_folder_type import get_simulation_folder_type
from copy_results import copy_results
from rate_target_size import calculate_radius
from database_helper import initialize_db, get_completed_simulations, insert_params, update_rating
from plot_FRD import plot_frequency_responses  # import the plotting function

# ------------------------------
# USER-CONFIGURABLE THRESHOLD
# ------------------------------
# Any simulation (old or new) with a rating above this is considered invalid.
# Old simulations above this threshold will be skipped.
# New simulations that compute a rating above this threshold will be **clipped** to THRESHOLD_RATING.
THRESHOLD_RATING = 999.0

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
RESULTS_FOLDER = config["Paths"]["RESULTS_FOLDER"]
HORNS_FOLDER = config["Paths"]["HORNS_FOLDER"]
ATH_EXE_PATH = config["Paths"]["ATH_EXE_PATH"]
CONFIGS_FOLDER = config["Paths"]["CONFIGS_FOLDER"]

# Initialize the database (clear it beforehand when reconfiguring parameters)
initialize_db("waveguides.db")

# Prepare structures for fixed parameters, optimization space, and variable parameter names.
fixed_params = {}
space = []
variable_names = []

def add_param(param_name, lower=None, upper=None):
    if lower == upper:
        fixed_params[param_name] = lower
    else:
        space.append(Real(lower, upper, name=param_name))
        variable_names.append(param_name)

# Populate fixed parameters and optimization space.
add_param('r0', 14.0, 14.0)
add_param('L', 15.0, 40.0)
add_param('a0', 0, 60.0)
add_param('a', 40.0, 75.0)
add_param('k', 0.0, 10.0)
add_param('s', 0.0, 2.0)
add_param('q', 0.99, 1.0)
add_param('n', 2.0, 10.0)
add_param('va', 20.0, 20.0)
add_param('mfp', -10.0, 0.0) #using this as ZOFF here!
add_param('mr', 2.0, 2.0)
add_param('u_va0', 0.0, 1.0)
add_param('u_vk', 0.0, 1.0)
add_param('u_vs', 0.0, 1.0)
add_param('u_vn', 0.0, 1.0)

target_size = 65

# -----------------------------------------------------------
# 1) LOAD PREVIOUS SIMULATIONS, IGNORING THOSE ABOVE THRESHOLD
# -----------------------------------------------------------
simulations, old_ratings = get_completed_simulations(db_path="waveguides.db")
print(f"Successfully loaded {len(simulations)} previous simulation result(s) from the database.")

x0_filtered = []
y0_filtered = []
skipped_count = 0

for sim_dict, rating in zip(simulations, old_ratings):
    if rating <= THRESHOLD_RATING:
        x_vector = [sim_dict[name] for name in variable_names]
        x0_filtered.append(x_vector)
        y0_filtered.append(rating)
    else:
        skipped_count += 1
        print(f"Skipping old waveguide with rating {rating} above threshold {THRESHOLD_RATING}.")

print(f"Using {len(x0_filtered)} data points after skipping {skipped_count} due to threshold.")
x0, y0 = x0_filtered, y0_filtered

# Function to create a marker file for failed waveguides.
def create_marker_file(results_folder, foldername, rating):
    marker_filename = f"{rating:.2f}_{foldername}.png"
    marker_file_path = os.path.join(results_folder, marker_filename)
    with open(marker_file_path, "w") as marker_file:
        marker_file.write(f"Rating: {rating}\n")
        marker_file.write(f"Foldername: {foldername}\n")
    print(f"Created marker file for failed waveguide: {marker_filename}")

# -----------------------------------------------------------
# OBJECTIVE FUNCTION
# -----------------------------------------------------------
def objective(params):
    # Unpack parameters and merge with fixed parameters.
    param_values = {dim.name: val for dim, val in zip(space, params)}
    param_values.update(fixed_params)

    formatted_params = {
        'r0': float(f"{param_values['r0']:.2f}"),
        'L': float(f"{param_values['L']:.2f}"),
        'a': float(f"{param_values['a']:.2f}"),
        'a0': float(f"{param_values['a0']:.2f}"),
        'k': float(f"{param_values['k']:.2f}"),
        's': float(f"{param_values['s']:.2f}"),
        'q': float(f"{param_values['q']:.3f}"),
        'n': float(f"{param_values['n']:.2f}"),
        'va': float(f"{param_values['va']:.2f}"),
        'u_va0': float(f"{param_values['u_va0']:.3f}"),
        'u_vk': float(f"{param_values['u_vk']:.3f}"),
        'u_vs': float(f"{param_values['u_vs']:.3f}"),
        'u_vn': float(f"{param_values['u_vn']:.3f}"),
        'mfp': float(f"{param_values['mfp']:.2f}"),
        'mr': float(f"{param_values['mr']:.2f}")
    }

    # --- Database Integration: Insert new simulation data.
    config_id = insert_params(formatted_params, db_path="waveguides.db")
    filename = f"{config_id}.cfg"
    foldername = str(config_id)

    if not generate_waveguide_config(CONFIGS_FOLDER, filename, config_id, verbose=False):
        print("Failed to generate config.")
        return 1e6

    sim_folder, sim_type = get_simulation_folder_type("base_template.txt")
    if not generate_abec_file(CONFIGS_FOLDER, ATH_EXE_PATH, filename, verbose=False):
        print("Failed to generate ABEC file.")
        return 1e6
    if not run_abec_simulation(foldername):
        print("Failed to run ABEC simulation.")
        return 1e6
    if not generate_report(CONFIGS_FOLDER, ATH_EXE_PATH, filename, verbose=True):
        print("Failed to generate report.")
        return 1e6

    # Compute the FR rating.
    try:
        fr_rating = rate_frequency_response(HORNS_FOLDER, foldername, sim_folder, verbose=False)
        print(f"FR Rating: {fr_rating}")
    except Exception as e:
        print(f"Error calculating FR rating: {e}")
        fr_rating = 1e4

    # Compute size penalty.
    try:
        radius = calculate_radius(
            formatted_params['a0'],
            formatted_params['a'],
            formatted_params['r0'],
            formatted_params['k'],
            formatted_params['L'],
            formatted_params['s'],
            formatted_params['n'],
            formatted_params['q']
        )
        size_penalty = abs(radius - target_size)
        print(f"Calculated radius: {radius:.2f}, Target: {target_size}, Penalty: {size_penalty:.2f}")
    except Exception as e:
        print(f"Error calculating radius: {e}")
        size_penalty = 1e6

    total_rating = fr_rating + size_penalty
    print(f"Total Rating: {total_rating}")

    # If new simulation rating exceeds the threshold, clip it to the threshold.
    if total_rating > THRESHOLD_RATING:
        print(f"New simulation rating {total_rating} exceeds threshold {THRESHOLD_RATING} => clipping to threshold.")
        total_rating = THRESHOLD_RATING+0.1

    # Update the new rating in the database.
    update_rating(config_id, total_rating, db_path="waveguides.db")

    # Plot the frequency response.
    try:
        plot_frequency_responses(
            foldername, sim_folder,
            HORNS_FOLDER, RESULTS_FOLDER,
            total_rating, config_id,
            formatted_params
        )
    except Exception as e:
        print(f"Error generating FRD plot: {e}")

    return total_rating

# -----------------------------------------------------------
# VALIDATE PREVIOUS DATA POINTS (OLD SIMULATIONS)
# -----------------------------------------------------------
invalid_points = []
for i, point in enumerate(x0):
    out_of_bounds = False
    out_of_bounds_params = {}
    for val, dim in zip(point, space):
        if not (dim.low <= val <= dim.high):
            out_of_bounds = True
            out_of_bounds_params[dim.name] = {"value": val, "bounds": (dim.low, dim.high)}
    if out_of_bounds:
        invalid_points.append((i, point, out_of_bounds_params))

if invalid_points:
    print(f"Found {len(invalid_points)} point(s) out of bounds in `x0`:")
    for index, point, params in invalid_points:
        print(f"\nPoint #{index} out of bounds: {point}")
        for param_name, info in params.items():
            print(f"  Parameter '{param_name}' = {info['value']} (out of bounds: {info['bounds']})")
else:
    print("All points in `x0` are within bounds.")

# -----------------------------------------------------------
# RUN BAYESIAN OPTIMIZATION (IF NO INVALID POINTS)
# -----------------------------------------------------------
if not invalid_points:
    if x0 and y0:
        print("Using valid old data points.")
        result = gp_minimize(
            func=objective,
            dimensions=space,
            x0=x0,
            y0=y0,
            n_calls=256,
            n_initial_points=128,
            acq_func="gp_hedge",
            acq_optimizer="auto",
            initial_point_generator='lhs',
            n_jobs=16,
            verbose=True,

        )
        optimal_params = result.x
        optimal_rating = result.fun
        print("Optimal parameters:", optimal_params)
        print("Optimal rating:", optimal_rating)
    else:
        print("No valid previous data points. Starting fresh optimization.")
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=256,
            n_initial_points=128,
            acq_func="gp_hedge",
            acq_optimizer="auto",
            initial_point_generator='sobol',
            n_jobs=16,
            verbose=True,

        )
        optimal_params = result.x
        optimal_rating = result.fun
        print("Optimal parameters:", optimal_params)
        print("Optimal rating:", optimal_rating)
else:
    print("Please adjust out-of-bounds points in `x0` to proceed with optimization.")
