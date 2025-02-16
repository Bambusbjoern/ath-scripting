import os
import sqlite3
import configparser

# Import functions from your modules.
from database_helper import get_params_by_id, update_rating
from rate_FRD import rate_frequency_response
from plot_FRD import plot_frequency_responses

# Load configuration from config.ini.
config = configparser.ConfigParser()
config.read("config.ini")
HORNS_FOLDER = config["Paths"]["HORNS_FOLDER"]
RESULTS_FOLDER = config["Paths"]["RESULTS_FOLDER"]
# For this script we assume the simulation folder type is constant:
SIMULATION_FOLDER = "ABEC_FreeStanding"
DB_PATH = "waveguides.db"

def clear_results_folder(results_folder):
    """
    Deletes all PNG files from the specified results folder.
    """
    if os.path.exists(results_folder):
        for filename in os.listdir(results_folder):
            file_path = os.path.join(results_folder, filename)
            if os.path.isfile(file_path) and filename.lower().endswith('.png'):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        print(f"Cleared all PNG files from {results_folder}")
    else:
        print(f"Results folder not found: {results_folder}")

def get_all_simulation_ids(db_path):
    """
    Retrieves all simulation IDs from the waveguides database.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id FROM waveguide_params")
    ids = [row[0] for row in c.fetchall()]
    conn.close()
    return ids

def delete_simulation(config_id, db_path=DB_PATH):
    """
    Deletes the simulation with the given config_id from the database.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM waveguide_params WHERE id=?", (config_id,))
    conn.commit()
    conn.close()
    print(f"Deleted simulation {config_id} from the database.")

def update_ratings_for_all_simulations():
    """
    Iterates over all finished simulations in the horns folder, recalculates the rating
    using rate_frequency_response, plots the frequency response with plot_frequency_responses,
    and updates the rating in the database. If a simulation’s rating is 10000 (indicating a bug),
    that simulation is deleted from the database.
    """
    # First, clear out any previous PNG plots.
    clear_results_folder(RESULTS_FOLDER)

    simulation_ids = get_all_simulation_ids(DB_PATH)
    if not simulation_ids:
        print("No simulations found in the database.")
        return

    for config_id in simulation_ids:
        foldername = str(config_id)  # Simulation folder is named as the config ID.
        # Retrieve parameters for this simulation.
        params_dict = get_params_by_id(config_id, db_path=DB_PATH)
        if params_dict is None:
            print(f"Simulation {config_id} not found in database.")
            continue

        # Recalculate the frequency response rating.
        try:
            new_rating = rate_frequency_response(HORNS_FOLDER, foldername, SIMULATION_FOLDER, verbose=False)
        except Exception as e:
            print(f"Error rating simulation {config_id}: {e}")
            new_rating = 10000  # Use 10000 as the flag for failure.

        print(f"Simulation {config_id}: New FR rating = {new_rating}")

        # If the new rating indicates a buggy simulation (10000), delete this simulation from the DB.
        if new_rating == 10000:
            print(f"Simulation {config_id} has a rating of {new_rating} (buggy). Deleting from database.")
            delete_simulation(config_id, db_path=DB_PATH)
            continue

        # Otherwise, update the new rating in the database.
        update_rating(config_id, new_rating, db_path=DB_PATH)

        # Plot the frequency response.
        try:
            # The plot function expects:
            # - config_number: here we use foldername (which is config_id as a string),
            # - SIMULATION_FOLDER, HORNS_FOLDER, RESULTS_FOLDER, new_rating, config_id, and the parameter dictionary.
            plot_frequency_responses(foldername, SIMULATION_FOLDER, HORNS_FOLDER, RESULTS_FOLDER, new_rating, config_id, params_dict)
        except Exception as e:
            print(f"Error plotting simulation {config_id}: {e}")

if __name__ == "__main__":
    update_ratings_for_all_simulations()
