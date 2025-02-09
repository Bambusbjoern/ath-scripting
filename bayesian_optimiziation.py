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

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read("config.ini")
RESULTS_FOLDER = config["Paths"]["RESULTS_FOLDER"]
HORNS_FOLDER = config["Paths"]["HORNS_FOLDER"]
ATH_EXE_PATH = config["Paths"]["ATH_EXE_PATH"]
CONFIGS_FOLDER = config["Paths"]["CONFIGS_FOLDER"]

# Initialize the database (clear it beforehand when reconfiguring parameters)
initialize_db("waveguides.db")

# Initialize dictionaries for fixed parameters, optimization space, and variable parameter names
fixed_params = {}
space = []
variable_names = []  # This will hold the names of parameters that are variable
# (No need to initialize x0, y0 here, they will be loaded from the database)

# Define parameter bounds directly and categorize parameters as fixed or variable
def add_param(param_name, lower=None, upper=None):
    if lower == upper:
        # Treat as fixed if the bounds are identical
        fixed_params[param_name] = lower
    else:
        # Add to the optimization space and record as variable
        space.append(Real(lower, upper, name=param_name))
        variable_names.append(param_name)

# Populate fixed parameters and optimization space.
add_param('r0', 14.0, 14.0)
add_param('L', 15.0, 40.0)
add_param('a0', 0.0, 60.0)
add_param('a', 40.0, 75.0)
add_param('k', 0.0, 10.0)
add_param('s', 0.0, 2.0)
add_param('q', 0.99, 1.0)
add_param('n', 2.0, 10.0)
add_param('va', 20, 20)
add_param('mfp', 0.0, 0.0)
add_param('mr', 1.0, 10.0)
add_param('u_va0', 0.0, 1.0)
add_param('u_vk', 0.0, 1.0)
add_param('u_vs', 0.0, 1.0)
add_param('u_vn', 0.0, 1.0)

target_size = 65

# Load previous simulation results from the database.
# (Make sure your database_helper.get_completed_simulations function returns a list of dictionaries.)
simulations, y0 = get_completed_simulations(db_path="waveguides.db")
print(f"Successfully loaded {len(simulations)} previous simulation result(s) from the database")

# Build the optimization vector x0 using the current variable_names order.
# This will extract values in the order defined by variable_names.
x0 = [[sim[name] for name in variable_names] for sim in simulations]

# (If there are no previous simulations, x0 may be empty. That is fine.)

# Function to create a marker file for failed waveguides
def create_marker_file(results_folder, foldername, rating):
    marker_filename = f"{rating:.2f}_{foldername}.png"
    marker_file_path = os.path.join(results_folder, marker_filename)
    with open(marker_file_path, "w") as marker_file:
        marker_file.write(f"Rating: {rating}\n")  # Optionally include additional details
        marker_file.write(f"Foldername: {foldername}\n")
    print(f"Created marker file for failed waveguide: {marker_filename}")

# Inside the objective function
def objective(params):
    # Unpack variable parameters from the optimizer and merge with fixed parameters.
    param_values = {name: val for name, val in zip([d.name for d in space], params)}
    param_values.update(fixed_params)  # Add the fixed parameters

    # Format parameters with the desired precision.
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

    # --- Database Integration ---
    # Insert the new parameter set into the database and get its configuration ID.
    config_id = insert_params(formatted_params, db_path="waveguides.db")
    filename = f"{config_id}.cfg"  # Use the ID as the filename
    foldername = str(config_id)      # And also as the folder name

    # Generate the configuration file.
    if not generate_waveguide_config(CONFIGS_FOLDER, filename, config_id, verbose=False):
        print("Failed to generate config.")
        return 1e6
    # --- End Database Integration ---

    # Retrieve simulation folder and type (from your template).
    simulation_folder, sim_type = get_simulation_folder_type("base_template.txt")

    if not generate_abec_file(CONFIGS_FOLDER, ATH_EXE_PATH, filename, verbose=False):
        print("Failed to generate ABEC file.")
        return 1e6

    if not run_abec_simulation(foldername):
        print("Failed to run ABEC simulation.")
        return 1e6

    if not generate_report(CONFIGS_FOLDER, ATH_EXE_PATH, filename, verbose=True):
        print("Failed to generate report.")
        return 1e6

    # Initialize ratings with a high penalty by default.
    fr_rating = 1e4

    try:
        # Calculate the FR rating from the FRD folder data.
        fr_rating = rate_frequency_response(HORNS_FOLDER, foldername, simulation_folder, verbose=False)
        print(f"FR Rating: {fr_rating}")
    except Exception as e:
        print(f"Error calculating FR rating: {e}")

    # Calculate the size deviation penalty.
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

    # Combine ratings to compute the total rating.
    total_rating = fr_rating + size_penalty  # + di_rating (DI is currently commented out)
    print(f"Total Rating: {total_rating}")

    # Update the rating in the database.
    update_rating(config_id, total_rating, db_path="waveguides.db")

    # Call the plot_FRD function to generate and save the plot.
    try:
        # We pass the following:
        # - The configuration number for plotting (here we use foldername, which equals config_id)
        # - simulation_folder, HORNS_FOLDER, RESULTS_FOLDER, total_rating, config_id, and the parameter dictionary.
        plot_frequency_responses(foldername, simulation_folder, HORNS_FOLDER, RESULTS_FOLDER,
                                 total_rating, config_id, formatted_params)
    except Exception as e:
        print(f"Error generating FRD plot: {e}")

    return total_rating


# Validate previous simulation results (x0) against the optimization space.
invalid_points = []
for i, point in enumerate(x0):
    out_of_bounds = False
    out_of_bounds_params = {}
    for param_value, dimension in zip(point, space):
        if not (dimension.low <= param_value <= dimension.high):
            out_of_bounds = True
            out_of_bounds_params[dimension.name] = {
                "value": param_value,
                "bounds": (dimension.low, dimension.high)
            }
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

# Proceed with Bayesian optimization only if all points are valid.
if not invalid_points:
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=384,
        n_initial_points=256,
        acq_func="gp_hedge",
        acq_optimizer="auto",
        initial_point_generator='sobol',
        verbose=True,
        n_jobs=-1
    )
    optimal_params = result.x
    optimal_rating = result.fun
    print("Optimal parameters:", optimal_params)
    print("Optimal rating:", optimal_rating)
else:
    print("Please adjust out-of-bounds points in `x0` to proceed with optimization.")
