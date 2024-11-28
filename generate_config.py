import os

def generate_waveguide_config(configs_folder, r0, a0, a, k, L, s, n, q, va, va0, vk, vs, vn, vq, mfp, mr, verbose=False):
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
    - va0: Additional angle adjustment for a0.
    - vk: Adjustment for k.
    - vs: Adjustment for s.
    - vn: Adjustment for n.
    - vq: Adjustment for q.
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

        # Generate config content
        config_content = base_template_content.format(
            r0=r0, a0=a0, a=a, k=k, L=L, s=s, n=n, q=q,
            va=va, va0=va0, vk=vk, vs=vs, vn=vn, vq=vq, mfp=mfp, mr=mr
        )
        
        # Define the filename based on parameters (without negative signs in values)
        filename = (
            f"L_{L:.2f}_a_{a:.2f}_r0_{r0:.2f}_a0_{a0:.2f}_k_{k:.2f}_s_{s:.2f}_q_{q:.3f}_n_{n:.2f}"
            f"_va_{va:.2f}_va0_{va0:.2f}_vk_{vk:.2f}_vs_{vs:.2f}_vn_{vn:.2f}_vq_{vq:.3f}_mfp_{mfp:.2f}_mr_{mr:.2f}.cfg"
        )
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
