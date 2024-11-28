import os


def generate_waveguide_config(configs_folder, r0, a0, a, k, L, s, n, q, va, u_va0, u_vk, u_vs, u_vn, mfp, mr, verbose=False):
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
        va0 = -(- a0 + u_va0 * 60) # Derived parameter for va0
        vk = -(-k + u_vk * 10)     # Derived parameter for vk
        vs = -(-s + u_vs * 2)      # Derived parameter for vs
        vn = -(2 - n + u_vn * 8)  # Derived parameter for vn

        # Generate config content
        config_content = base_template_content.format(
            r0=r0, a0=a0, a=a, k=k, L=L, s=s, n=n, q=q,
            va=va, va0=va0, vk=vk, vs=vs, vn=vn, mfp=mfp, mr=mr
        )

        # Define the filename based on normalized parameters (u_va0, u_vk, u_vs, u_vn)
        filename = (
            f"L{L:.1f}A{a:.0f}R{r0:.0f}"
            f"A0{a0:.1f}K{k:.1f}S{s:.1f}"
            f"Q{q:.3f}N{n:.1f}UVA0{u_va0:.3f}"
            f"UVK{u_vk:.3f}UVS{u_vs:.3f}UVN{u_vn:.3f}"
            f"M{mfp:.2f}MR{mr:.1f}.cfg"
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
