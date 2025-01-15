import numpy as np

# Function to calculate waveguide size based on given parameters
def calculate_radius(a0_deg, a_deg, r0, k, L, s, n, q):
    a0 = np.radians(a0_deg)
    a = np.radians(a_deg)
    x = L
    term1 = np.sqrt((k * r0) ** 2 + 2 * k * r0 * x * np.tan(a0) + (x * np.tan(a)) ** 2)
    term2 = r0 * (1 - k)
    term3 = (L * s / q) * (1 - (1 - (q * x / L) ** n) ** (1 / n))
    return term1 + term2 + term3
