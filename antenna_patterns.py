"""
antenna_patterns.py

A collection of simplified, normalized far-field radiation pattern formulas 
for popular antenna topologies.

Coordinate System:
- r: radial distance
- theta: polar/elevation angle (0 is the +z-axis, pi is the -z-axis)
- phi: azimuthal angle (0 is the +x-axis, pi/2 is the +y-axis)
"""

import numpy as np

# A tiny value to prevent division by zero in numpy arrays
EPSILON = 1e-12

def hertzian_dipole(theta):
    """
    Computes the normalized pattern of an infinitesimal Hertzian dipole 
    aligned with the z-axis.
    """
    return np.abs(np.sin(theta))

def half_wave_dipole(theta):
    """
    Computes the normalized pattern of a half-wave dipole 
    aligned with the z-axis.
    """
    numerator = np.cos((np.pi / 2.0) * np.cos(theta))
    denominator = np.sin(theta) + EPSILON
    return np.abs(numerator / denominator)

def quarter_wave_monopole(theta):
    """
    Computes the normalized pattern of a quarter-wave monopole 
    mounted vertically on an infinite ground plane (x-y plane).
    """
    pattern = half_wave_dipole(theta)
    # Zero out radiation below the ground plane (theta > pi/2)
    pattern = np.where(theta <= np.pi / 2.0, pattern, 0.0)
    return pattern

def patch_e_plane(theta, freq_hz, L_e):
    """
    Computes the normalized E-plane pattern (x-z plane, phi=0) 
    of a rectangular microstrip patch.
    """
    c = 299792458.0 # Speed of light in m/s
    lambda_0 = c / freq_hz
    k_0 = (2.0 * np.pi) / lambda_0
    
    return np.abs(np.cos((k_0 * L_e / 2.0) * np.sin(theta)))

def patch_h_plane(theta, freq_hz, W):
    """
    Computes the normalized H-plane pattern (y-z plane, phi=90) 
    of a rectangular microstrip patch.
    """
    c = 299792458.0
    lambda_0 = c / freq_hz
    k_0 = (2.0 * np.pi) / lambda_0
    
    arg = (k_0 * W / 2.0) * np.sin(theta)
    # Use sinc function (np.sinc is normalized as sin(pi*x)/(pi*x), so we adjust)
    sinc_term = np.sin(arg) / (arg + EPSILON) 
    
    return np.abs(np.cos(theta) * sinc_term)

def uniform_linear_array(theta, N, d_lambda, beta_radians):
    """
    Computes the Array Factor (AF) for a Uniform Linear Array (ULA)
    aligned along the z-axis.
    
    Parameters:
    - N: Number of elements
    - d_lambda: Spacing between elements in wavelengths
    - beta_radians: Progressive phase shift between elements in radians
    """
    psi = (2.0 * np.pi * d_lambda * np.cos(theta)) + beta_radians
    
    numerator = np.sin(N * psi / 2.0)
    denominator = N * np.sin(psi / 2.0) + EPSILON
    
    return np.abs(numerator / denominator)

def yagi_uda_approx(theta, N, currents_mag, currents_phase, z_positions_lambda):
    """
    Computes the approximate normalized pattern of a Yagi-Uda antenna using 
    pattern multiplication (assuming all elements approximate a half-wave dipole).
    
    Parameters:
    - N: Total number of elements
    - currents_mag: List/array of current magnitudes for each element
    - currents_phase: List/array of current phases (in radians) for each element
    - z_positions_lambda: List/array of element positions along the boom (z-axis) in wavelengths
    """
    # Base element pattern
    element_pattern = half_wave_dipole(theta)
    
    # Complex Array Factor calculation
    array_factor = np.zeros_like(theta, dtype=complex)
    for n in range(N):
        phase_term = (2.0 * np.pi * z_positions_lambda[n] * np.cos(theta)) + currents_phase[n]
        array_factor += currents_mag[n] * np.exp(1j * phase_term)
        
    pattern = element_pattern * np.abs(array_factor)
    
    # Normalize the output
    max_val = np.max(pattern)
    return pattern / (max_val + EPSILON)

def helical_axial(theta, N, S_lambda):
    """
    Computes the normalized pattern for an axial-mode Helical antenna 
    using the Hansen-Woodyard condition for optimal end-fire directivity.
    
    Parameters:
    - N: Number of turns
    - S_lambda: Spacing between turns in wavelengths
    """
    # Single turn approximation
    turn_pattern = np.cos(theta)
    turn_pattern = np.where(turn_pattern > 0, turn_pattern, 0) # Only radiates forward
    
    # Hansen-Woodyard phase condition
    psi = (2.0 * np.pi * S_lambda * (np.cos(theta) - 1.0)) - (np.pi / N)
    
    # Array factor for the loops
    numerator = np.sin(N * psi / 2.0)
    denominator = N * np.sin(psi / 2.0) + EPSILON
    array_factor = np.abs(numerator / denominator)
    
    return np.abs(turn_pattern * array_factor)

def eggbeater(theta, phi):
    """
    Computes the simplified total magnitude pattern for an Eggbeater antenna 
    (crossed full-wave loops over a lambda/4 ground plane).
    """
    # Note: Precise full-wave loop patterns are complex Bessel functions. 
    # For a simplified model, we approximate the loops as broad cosine lobes.
    # Loop X sits in the x-z plane
    loop_x = np.cos(theta) * np.cos(phi)
    # Loop Y sits in the y-z plane
    loop_y = np.cos(theta) * np.sin(phi)
    
    # Crossed loops fed 90 degrees (pi/2) out of phase
    quadrature_loops = np.abs(loop_x + loop_y * np.exp(-1j * np.pi / 2.0))
    
    # Ground plane image factor (array factor for an element lambda/4 above ground)
    # Height h = lambda/4, so k*h = (2*pi/lambda)*(lambda/4) = pi/2
    ground_factor = np.abs(np.sin((np.pi / 2.0) * np.cos(theta)))
    
    pattern = quadrature_loops * ground_factor
    
    # Normalize
    max_val = np.max(pattern)
    return pattern / (max_val + EPSILON)