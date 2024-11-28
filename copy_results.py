import os
import shutil

def copy_results(png_file_path, foldername, total_rating, results_folder, verbose=False):
    """
    Copies the PNG file to the results folder with a rating prefix.

    Args:
        png_file_path (str): Path to the PNG file.
        foldername (str): Name of the folder being processed.
        total_rating (float): Calculated total rating for the simulation.
        results_folder (str): Destination folder for results.
        verbose (bool): Whether to print verbose output.
    """
    try:
        # Construct the new PNG file name with the rating prefix
        rating_str = f"{total_rating:.2f}"
        new_png_name = f"{rating_str}_{foldername}.png"
        new_png_path = os.path.join(results_folder, new_png_name)

        # Copy the PNG file
        shutil.copy(png_file_path, new_png_path)

        if verbose:
            print(f"Copied PNG file to: {new_png_path}")
    except Exception as e:
        print(f"Error copying file: {e}")
