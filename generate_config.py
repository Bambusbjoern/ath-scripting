import os


def generate_waveguide_config(configs_folder, filename, r0, a0, a, k, L, s, n, q, va, u_va0, u_vk, u_vs, u_vn, mfp, mr, verbose=False):
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
    - va: Fixed parameter for vertical angle.
    - u_va0: Normalized parameter for va0.
    - u_vk: Normalized parameter for vk.
    - u_vs: Normalized parameter for vs.
    - u_vn: Normalized parameter for vn.
    - mfp: Morphing fixed part.
    - mr: Morphing rate.
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

        # Calculate derived parameters
        va0 = round(-(- a0 + u_va0 * 60),2) # Derived parameter for va0
        vk = round(-(-k + u_vk * 10),2)     # Derived parameter for vk
        vs = round(-(-s + u_vs * 2),2)      # Derived parameter for vs
        vn = round(-(2 - n + u_vn * 8),2)  # Derived parameter for vn

        # Generate config content
        config_content = base_template_content.format(
            r0=r0, a0=a0, a=a, k=k, L=L, s=s, n=n, q=q,
            va=va, va0=va0, vk=vk, vs=vs, vn=vn, mfp=mfp, mr=mr
        )


        # Generate the folder name if needed
        foldername = filename.replace('.cfg', '')

        # Combine with the configs folder
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
