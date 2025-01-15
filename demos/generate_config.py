import os

def generate_waveguide_config(configs_folder, r0, a0, a, k, L, s, n, q, verbose=False):
    """
    Generates a waveguide configuration file based on provided parameters.

    Parameters:
    - configs_folder (str): Path to the folder where configuration files should be saved.
    - r0: Radius parameter.
    - a0: Angle a0 in degrees.
    - a: Angle a in degrees.
    - k: Constant k.
    - L: Length L.
    - s: Parameter s.
    - n: Parameter n.
    - q: Parameter q.
    - verbose (bool): If True, print additional debug information.

    Returns:
    - True if the configuration file was created successfully, False otherwise.
    """
    try:
        # Ensure the output directory exists
        if not os.path.exists(configs_folder):
            os.makedirs(configs_folder)
            if verbose:
                print(f"Created directory: {configs_folder}")

        # Load the base template content
        with open('base_template.txt', 'r') as file:
            base_template_content = file.read()

        # Generate config content
        config_content = base_template_content.format(r0=r0, a0=a0, a=a, k=k, L=L, s=s, n=n, q=q)
        
        # Define the filename based on parameters
        filename = f"L-{L:.2f}_a-{a:.2f}_r0-{r0:.2f}_a0-{a0:.2f}_k-{k:.2f}_s-{s:.2f}_q-{q:.3f}_n-{n:.2f}.cfg"
        filepath = os.path.join(configs_folder, filename)

        # Write the configuration to a file
        with open(filepath, 'w') as config_file:
            config_file.write(config_content)
        
        if verbose:
            print(f"Created configuration file: {filepath}")
        return True
    except Exception as e:
        if verbose:
            print(f"Failed to create configuration file: {e}")
        return False
