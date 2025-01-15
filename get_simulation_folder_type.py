import sys
# Function to determine simulation folder based on ABEC.SimType in base_template.txt
def get_simulation_folder_type(template_path="base_template.txt"):
    try:
        with open(template_path, "r") as file:
            for line in file:
                # Remove any comment after the `;` character
                line = line.split(";")[0].strip()
                
                # Look for the line defining `ABEC.SimType`
                if line.startswith("ABEC.SimType"):
                    sim_type = int(line.split("=")[1].strip())
                    folder = "ABEC_InfiniteBaffle" if sim_type == 1 else "ABEC_FreeStanding"
                    return folder, sim_type
                    
        # If we complete the loop and don't find ABEC.SimType, raise an error
        raise ValueError("ABEC.SimType not found in the template.")
    
    except FileNotFoundError:
        print(f"Template file not found: {template_path}")
        sys.exit("Exiting: Please ensure the template file is available.")
    except ValueError as e:
        print(f"Error reading simulation type from template: {e}")
        sys.exit("Exiting: Please check the format of ABEC.SimType in the template file.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit("Exiting: An unexpected error occurred.")