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
from rate_DI import rate_DI
from rate_FR import rate_frequency_response
from get_simulation_folder_type import get_simulation_folder_type
from copy_results import copy_results
from rate_target_size import calculate_radius
from database_helper import initialize_db, get_completed_simulations, insert_params, update_rating


# Load configuration to get the results folder path
config = configparser.ConfigParser()
config.read("config.ini")
RESULTS_FOLDER = config["Paths"]["RESULTS_FOLDER"]
HORNS_FOLDER = config["Paths"]["HORNS_FOLDER"]
ATH_EXE_PATH = config["Paths"]["ATH_EXE_PATH"]
CONFIGS_FOLDER = config["Paths"]["CONFIGS_FOLDER"]

initialize_db("waveguides.db")

# Initialize lists for fixed parameters, optimization space, and variable parameter names
fixed_params = {}
space = []
variable_names = []  # Track variable parameter names automatically
x0, y0 = [], []  # Initialize lists for previous parameters and ratings

# Define parameter bounds directly and categorize parameters as fixed or variable
def add_param(param_name, lower=None, upper=None):
    if lower == upper:
        # Treat as fixed if identical bounds
        fixed_params[param_name] = lower
    else:
        # Add to the optimization space and record as variable
        space.append(Real(lower, upper, name=param_name))
        variable_names.append(param_name)

# Populate fixed parameters and optimization space
add_param('r0', 14.0, 14.0) # Fixed to 14
add_param('L', 15.0, 40.0)  # Optimized between 20 and 50
add_param('a0', 0.0, 60.0) # Optimized between 25 and 45
add_param('a', 60.0, 60.0)  # Fixed to 55
add_param('k', 0.0, 10.0)   # Optimized between 0.0 and 10
add_param('s', 0.0, 2.0)    # Optimized between 0.0 and 2
add_param('q', 0.99, 1.0)   # Optimized between 0.99 and 1.0
add_param('n', 2.0, 10.0)   # Optimized between 2.0 and 10.0
add_param('va', 20, 20)
add_param('mfp', 0.0, 0.0) #morph.fixedpart
add_param('mr', 1.0, 10.0)  #morph.rate
add_param('u_va0', 0.0, 1.0)  # Normalized range for va0
add_param('u_vk', 0.0, 1.0)   # Normalized range for vk
add_param('u_vs', 0.0, 1.0)   # Normalized range for vs
add_param('u_vn', 0.0, 1.0)   # Normalized range for vn

target_size = 65



# Ergebnisse aus der Datenbank laden
x0, y0 = get_completed_simulations(db_path="waveguides.db")
print(f"Successfully loaded {len(x0)} previous simulation result(s) from the database")


# Function to create a marker file for failed waveguides
def create_marker_file(results_folder, foldername, rating):
    marker_filename = f"{rating:.2f}_{foldername}.png"
    marker_file_path = os.path.join(results_folder, marker_filename)
    with open(marker_file_path, "w") as marker_file:
        marker_file.write(f"Rating: {rating}\n")  # Optionally include additional details in the file
        marker_file.write(f"Foldername: {foldername}\n")  # Include the foldername for reference
    print(f"Created marker file for failed waveguide: {marker_filename}")

# Inside the `objective` function
def objective(params):
    # Unpack parameters and add fixed parameters
    param_values = {name: val for name, val in zip([d.name for d in space], params)}
    param_values.update(fixed_params)  # Add fixed parameters

    # Format parameters to match the specified decimal precision
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

    # --- Hier wird der Datenbankteil integriert ---
    # Füge den Datensatz in die Datenbank ein und erhalte die neue ID
    config_id = insert_params(formatted_params, db_path="waveguides.db")
    # Verwende die ID als Dateinamen, z.B. "123.cfg"
    filename = f"{config_id}.cfg"
    foldername = str(config_id)  # Optional: Nutze die ID auch als Ordnername

    # Rufe die Funktion zum Erzeugen der Konfiguration auf und übergebe die ID
    if not generate_waveguide_config(
        CONFIGS_FOLDER, filename, config_id, verbose=False
    ):
        print("Failed to generate config.")
        return 1e6
    # --- Ende Datenbankintegration ---


    # Retrieve simulation folder and type
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


    # Initialize ratings
    di_rating = 1e6  # High penalty by default
    fr_rating = 1e6  # High penalty by default
    
    try:
        # Calculate FR rating
        fr_rating = rate_frequency_response(HORNS_FOLDER, foldername, simulation_folder, verbose=False)
        print(f"FR Rating: {fr_rating}")
    except Exception as e:
        print(f"Error calculating FR rating: {e}")

    try:
        # Calculate DI rating
        di_rating = rate_DI(HORNS_FOLDER, foldername, simulation_folder, verbose=False)
        print(f"DI Rating: {di_rating}")
    except Exception as e:
        print(f"Error calculating DI rating: {e}")


    # Calculate size deviation penalty
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
        size_penalty = 1e6  # High penalty for errors

    # Combine DI rating with other ratings
    total_rating = di_rating + fr_rating + size_penalty
    print(f"Total Rating: {total_rating}")


    update_rating(config_id, total_rating, db_path="waveguides.db")

    # Attempt to copy results
    try:
        png_file_path = os.path.join(HORNS_FOLDER, foldername, simulation_folder, "Results", f"{foldername}.png")
        if os.path.exists(png_file_path):
            copy_results(png_file_path, foldername, total_rating, RESULTS_FOLDER, verbose=True)
        else:
            print(f"PNG file not found: {png_file_path}")
    except Exception as e:
        print(f"Error copying results: {e}")

    return total_rating


# Function to calculate waveguide size based on given parameters
def calculate_y_at_L(a0_deg, a_deg, r0, k, L, s, n, q):
    a0 = np.radians(a0_deg)
    a = np.radians(a_deg)
    x = L
    term1 = np.sqrt((k * r0) ** 2 + 2 * k * r0 * x * np.tan(a0) + (x * np.tan(a)) ** 2)
    term2 = r0 * (1 - k)
    term3 = (L * s / q) * (1 - (1 - (q * x / L) ** n) ** (1 / n))
    return term1 + term2 + term3
    
# Check if there are any previous results; if not, start fresh
# Check if there are any points in `x0` that fall outside the bounds of `space`
invalid_points = []
for i, point in enumerate(x0):
    out_of_bounds = False
    out_of_bounds_params = {}  # Dictionary to store which parameters are out of bounds
    
    # Check each parameter in the point against its corresponding dimension in `space`
    for param_value, dimension in zip(point, space):
        if not (dimension.low <= param_value <= dimension.high):
            # If out of bounds, add the parameter name and value
            out_of_bounds = True
            out_of_bounds_params[dimension.name] = {
                "value": param_value,
                "bounds": (dimension.low, dimension.high)
            }
    
    # If any parameter was out of bounds, add to invalid_points
    if out_of_bounds:
        invalid_points.append((i, point, out_of_bounds_params))

# Print out-of-bounds points with detailed parameter information if any are found
if invalid_points:
    print(f"Found {len(invalid_points)} point(s) out of bounds in `x0`:")
    for index, point, params in invalid_points:
        print(f"\nPoint #{index} out of bounds: {point}")
        for param_name, info in params.items():
            print(f"  Parameter '{param_name}' = {info['value']} (out of bounds: {info['bounds']})")
else:
    print("All points in `x0` are within bounds.")

# Proceed with Bayesian optimization only if all points are valid
if not invalid_points:
    # Check if there are any previous results; if not, start fresh
    if x0 and y0:
        # Run Bayesian optimization with previous points
        result = gp_minimize(
            func=objective,            # Objective function to minimize
            dimensions=space,          # Only include parameters with ranges
            n_calls=512,               # Number of evaluations
            n_initial_points=256,      # Initial random evaluations before using model predictions
            acq_func="gp_hedge",       # Acquisition function, Expected Improvement
            acq_optimizer="auto",      # Automatically select optimizer for acquisition
            #n_restarts_optimizer=10,
            #random_state=42,            # Ensures reproducible results
            initial_point_generator='sobol',  # Use Sobol sampling for initial points
            verbose=True,
            n_jobs=-1
        )
    else:
        # Run Bayesian optimization without previous points
        result = gp_minimize(
            func=objective,            # Objective function to minimize
            dimensions=space,          # Only include parameters with ranges
            n_calls=512,               # Number of evaluations
            n_initial_points=256,      # Initial random evaluations before using model predictions
            acq_func="gp_hedge",       # Acquisition function, Expected Improvement
            acq_optimizer="auto",      # Automatically select optimizer for acquisition
            #n_restarts_optimizer=10,
            #random_state=42,            # Ensures reproducible results
            initial_point_generator='sobol',  # Use Sobol sampling for initial points
            verbose=True,
            n_jobs=-1
        )

    # Output the optimal parameters and rating
    optimal_params = result.x
    optimal_rating = result.fun  # The minimized rating

    print("Optimal parameters:", optimal_params)
    print("Optimal rating:", optimal_rating)
else:
    print("Please adjust out-of-bounds points in `x0` to proceed with optimization.")
    
