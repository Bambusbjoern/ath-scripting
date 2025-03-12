import os
import numpy as np
import configparser
import re
import time
import torch

# BoTorch imports for single-objective optimization.
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf

# Import your project modules.
from generate_config import generate_waveguide_config
from generate_abec_file import generate_abec_file
from run_abec import run_abec_simulation
from generate_report import generate_report
from rate_FRD import rate_frequency_response
from get_simulation_folder_type import get_simulation_folder_type
from rate_target_size import calculate_radius
from database_helper import initialize_db, get_completed_simulations, insert_params, update_rating
from plot_FRD import plot_frequency_responses  # plotting function
from skopt.space import Real

# ------------------------------
# USER-CONFIGURABLE THRESHOLD
# ------------------------------
THRESHOLD_RATING = 9999.0

# Load configuration from config.ini.
config = configparser.ConfigParser()
config.read("config.ini")
RESULTS_FOLDER = config["Paths"]["RESULTS_FOLDER"]
HORNS_FOLDER = config["Paths"]["HORNS_FOLDER"]
ATH_EXE_PATH = config["Paths"]["ATH_EXE_PATH"]
CONFIGS_FOLDER = config["Paths"]["CONFIGS_FOLDER"]

# Initialize the database.
initialize_db("waveguides.db")

# Set up device: use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
add_param('L', 10.0, 45.0)
add_param('a0', 20.0, 80.0)
add_param('a', 20.0, 80.0)
add_param('k', 0.1, 10.0)
add_param('s', 0.0, 2.0)
add_param('q', 0.99, 1.0)
add_param('n', 1.0, 10.0)
add_param('mfp', -4.0, -0.0)  # using this as ZOFF here!
add_param('mr', 2.0, 2.0)
add_param('u_va', 0.0, 1.0)
add_param('u_va0', 0.0, 1.0)
add_param('u_vk', 0.0, 1.0)
add_param('u_vs', 0.0, 1.0)
add_param('u_vn', 0.0, 1.0)

target_size = 65

# -----------------------------------------------------------
# LOAD PREVIOUS SIMULATIONS, IGNORING THOSE ABOVE THRESHOLD
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
if len(x0_filtered) > 0:
    X_train = torch.tensor(x0_filtered, dtype=torch.double).to(device)
    Y_train = torch.tensor(y0_filtered, dtype=torch.double).unsqueeze(-1).to(device)
else:
    X_train = torch.empty((0, len(space)), dtype=torch.double, device=device)
    Y_train = torch.empty((0, 1), dtype=torch.double, device=device)

# Create bounds tensor (in original domain): shape (2, d)
d = len(space)
bounds = torch.tensor([[dim.low for dim in space],
                       [dim.high for dim in space]], dtype=torch.double, device=device)

# ---- Scaling functions ----
def scale_point_torch(x, bounds):
    """Scales x (tensor) from original domain to [0,1]^d."""
    lower = bounds[0]
    upper = bounds[1]
    return (x - lower) / (upper - lower)

def unscale_point_torch(x_scaled, bounds):
    """Unscales x (tensor) from [0,1]^d to original domain."""
    lower = bounds[0]
    upper = bounds[1]
    return x_scaled * (upper - lower) + lower

# If we have previous data, scale it.
if X_train.shape[0] > 0:
    X_train = scale_point_torch(X_train, bounds)

# -----------------------------------------------------------
# OBJECTIVE FUNCTION (adapted for BoTorch with scaling)
# -----------------------------------------------------------
def objective(params):
    """
    Unpacks a parameter list (in original domain), runs the simulation, and returns the total rating.
    Returns a torch.tensor of shape (1,) in double precision.
    """
    # 'params' is in the original domain.
    param_values = {dim.name: val for dim, val in zip(space, params)}
    param_values.update(fixed_params)

    formatted_params = {
        'r0': float(f"{param_values['r0']:.3f}"),
        'L': float(f"{param_values['L']:.3f}"),
        'a0': float(f"{param_values['a0']:.3f}"),
        'a': float(f"{param_values['a']:.3f}"),
        'k': float(f"{param_values['k']:.3f}"),
        's': float(f"{param_values['s']:.3f}"),
        'q': float(f"{param_values['q']:.3f}"),
        'n': float(f"{param_values['n']:.3f}"),
        'u_va': float(f"{param_values['u_va']:.3f}"),
        'u_va0': float(f"{param_values['u_va0']:.3f}"),
        'u_vk': float(f"{param_values['u_vk']:.3f}"),
        'u_vs': float(f"{param_values['u_vs']:.3f}"),
        'u_vn': float(f"{param_values['u_vn']:.3f}"),
        'mfp': float(f"{param_values['mfp']:.3f}"),
        'mr': float(f"{param_values['mr']:.3f}")
    }

    config_id = insert_params(formatted_params, db_path="waveguides.db")
    filename = f"{config_id}.cfg"
    foldername_sim = str(config_id)

    if not generate_waveguide_config(CONFIGS_FOLDER, filename, config_id, verbose=False):
        print("Failed to generate config.")
        return torch.tensor([1e6], dtype=torch.double, device=device)

    sim_folder, sim_type = get_simulation_folder_type("base_template.txt")
    if not generate_abec_file(CONFIGS_FOLDER, ATH_EXE_PATH, filename, verbose=False):
        print("Failed to generate ABEC file.")
        return torch.tensor([1e6], dtype=torch.double, device=device)
    if not run_abec_simulation(foldername_sim):
        print("Failed to run ABEC simulation.")
        return torch.tensor([1e6], dtype=torch.double, device=device)
    if not generate_report(CONFIGS_FOLDER, ATH_EXE_PATH, filename, verbose=True):
        print("Failed to generate report.")
        return torch.tensor([1e6], dtype=torch.double, device=device)

    try:
        fr_rating = rate_frequency_response(HORNS_FOLDER, foldername_sim, sim_folder, verbose=False)
        print(f"FR Rating: {fr_rating}")
    except Exception as e:
        print(f"Error calculating FR rating: {e}")
        fr_rating = 1e4

    # Size penalty disabled.
    size_penalty = 0

    total_rating = fr_rating + size_penalty
    print(f"Total Rating: {total_rating}")

    if total_rating > THRESHOLD_RATING:
        print(f"New simulation rating {total_rating} exceeds threshold {THRESHOLD_RATING} => clipping to threshold.")
        total_rating = THRESHOLD_RATING + 0.1

    update_rating(config_id, total_rating, db_path="waveguides.db")
    try:
        plot_frequency_responses(
            foldername_sim, sim_folder,
            HORNS_FOLDER, RESULTS_FOLDER,
            total_rating, config_id,
            formatted_params
        )
    except Exception as e:
        print(f"Error generating FRD plot: {e}")

    return torch.tensor([total_rating], dtype=torch.double, device=device)

# -----------------------------------------------------------
# BO-TORCH OPTIMIZATION LOOP
# -----------------------------------------------------------
TOTAL_CALLS = 1024  # Adjusted for faster experimentation.
if X_train.shape[0] == 0:
    n_initial = 512
    # Generate n_initial points uniformly in [0,1]^d.
    sampler = SobolQMCNormalSampler(num)#torch.quasirandom.SobolEngine(d, scramble=True)
    X_init = sampler.draw(n_initial).to(device)
    X_train = X_init.clone()
    Y_list = []
    for x in X_train:
        x_orig = unscale_point_torch(x, bounds).tolist()
        y_val = objective(x_orig).item()
        Y_list.append(y_val)
    Y_train = torch.tensor(Y_list, dtype=torch.double, device=device).unsqueeze(-1)
    print(f"Generated {n_initial} initial design points.")
else:
    n_initial = X_train.shape[0]

n_iter = TOTAL_CALLS - n_initial
print(f"Starting optimization with {n_initial} initial points and {n_iter} iterations.")

# Optimization loop.
for i in range(n_iter):
    start_time = time.time()

    # Fit GP model on scaled data.
    model = SingleTaskGP(X_train, Y_train)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Use LogExpectedImprovement as the acquisition function.
    EI = LogExpectedImprovement(model=model, best_f=Y_train.min(), maximize=False)

    candidate_scaled, acq_value = optimize_acqf(
        acq_function=EI,
        bounds=torch.stack([torch.zeros(d, dtype=torch.double, device=device),
                            torch.ones(d, dtype=torch.double, device=device)]),
        q=1,
        num_restarts=20,  # Increased restarts
        raw_samples=50,  # More raw samples
        options={"batch_limit": 10, "maxiter": 300},  # Higher max iterations and batch limit
    )

    new_x_scaled = candidate_scaled.detach()
    new_x = unscale_point_torch(new_x_scaled, bounds)

    new_y_val = objective(new_x.cpu().numpy()[0]).item()
    new_y = torch.tensor([new_y_val], dtype=torch.double, device=device).unsqueeze(-1)

    X_train = torch.cat([X_train, new_x_scaled], dim=0)
    Y_train = torch.cat([Y_train, new_y], dim=0)

    end_time = time.time()
    iteration_time = end_time - start_time
    print(f"Iteration {i+1}: Candidate (unscaled) {new_x.cpu().numpy()[0]}, Objective {new_y_val},\nTime: {iteration_time:.2f} seconds")

best_index = torch.argmin(Y_train)
best_x_scaled = X_train[best_index]
best_x = unscale_point_torch(best_x_scaled, bounds)
best_y = Y_train[best_index].item()
print("Optimal parameters (BoTorch):", best_x.cpu().numpy())
print("Optimal rating:", best_y)
