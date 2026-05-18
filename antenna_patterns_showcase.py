"""
showcase.py

A visualization script to display the radiation patterns from antenna_patterns.py
Requires: matplotlib, numpy
Run: python showcase.py
"""

import numpy as np
import matplotlib.pyplot as plt
import antenna_patterns as ap

# ---------------------------------------------------------
# 1. Setup the Angle Arrays
# ---------------------------------------------------------
# For a standard 2D elevation cut, we want to plot a full circle.
# In a polar plot, 0 is typically zenith (+z axis). 
# We evaluate from -pi to +pi.
theta_plot = np.linspace(-np.pi, np.pi, 1000)

# Many formulas in the module assume spherical theta [0, pi].
# Because these elevation cuts are symmetrical around the z-axis, 
# we can pass the absolute value of our plotting angles.
theta_eval = np.abs(theta_plot)

# ---------------------------------------------------------
# 2. Calculate the Patterns
# ---------------------------------------------------------

# 1. Hertzian Dipole
p_hertzian = ap.hertzian_dipole(theta_eval)

# 2. Half-Wave Dipole
p_hw_dipole = ap.half_wave_dipole(theta_eval)

# 3. Quarter-Wave Monopole
p_monopole = ap.quarter_wave_monopole(theta_eval)

# 4. Microstrip Patch (2.4 GHz, L=40mm, W=50mm)
f_hz = 2.4e9
p_patch_e = ap.patch_e_plane(theta_eval, freq_hz=f_hz, L_e=0.04)
p_patch_h = ap.patch_h_plane(theta_eval, freq_hz=f_hz, W=0.05)

# 5. Uniform Linear Array (5 elements, half-wave spacing, broadside)
p_ula = ap.uniform_linear_array(theta_eval, N=5, d_lambda=0.5, beta_radians=0)

# 6. Yagi-Uda (Approximation of a 3-element: Reflector, Driven, Director)
# Elements placed along the z-axis (boom)
z_pos = [-0.25, 0.0, 0.2] # Wavelengths
mags = [0.9, 1.0, 0.8]    # Induced current magnitudes
phases = [-np.pi/4, 0, np.pi/4] # Induced phases
p_yagi = ap.yagi_uda_approx(theta_eval, N=3, currents_mag=mags, 
                            currents_phase=phases, z_positions_lambda=z_pos)

# 7. Helical Antenna (Axial Mode, 10 turns, quarter-wave pitch)
p_helical = ap.helical_axial(theta_eval, N=5, S_lambda=.25)

# 8. Eggbeater Antenna (Elevation slice at phi = 0)
p_eggbeater = ap.eggbeater(theta_eval, phi=0.0)

# ---------------------------------------------------------
# 3. Plotting Setup
# ---------------------------------------------------------
# Group the patterns and titles for easy iteration
patterns = [
    ("Hertzian Dipole", p_hertzian),
    ("Half-Wave Dipole", p_hw_dipole),
    ("1/4-Wave Monopole\n(Over Ground)", p_monopole),
    ("Patch Antenna\n(E-Plane)", p_patch_e),
    ("Patch Antenna\n(H-Plane)", p_patch_h),
    ("5-Element ULA\n(Broadside)", p_ula),
    ("3-Element Yagi-Uda\n(Approximation)", p_yagi),
    ("Helical Antenna\n(10 Turns, Axial)", p_helical),
    ("Eggbeater\n(Over Ground)", p_eggbeater)
]

fig = plt.figure(figsize=(14, 12))
fig.suptitle("Normalized Far-Field Antenna Radiation Patterns", fontsize=18, fontweight='bold', y=0.95)

for i, (title, pattern) in enumerate(patterns):
    # Create a 3x3 grid of polar subplots
    ax = fig.add_subplot(3, 3, i + 1, projection='polar')
    
    # Orient the polar plot so 0 degrees (Z-axis) is pointing straight UP
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1) # Clockwise
    
    # Plot the data
    ax.plot(theta_plot, pattern, color='b', linewidth=2)
    ax.fill(theta_plot, pattern, color='b', alpha=0.2)
    
    # Formatting
    ax.set_title(title, pad=15, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([]) # Hide radial ticks for a cleaner look
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0, 1, 0.93]) # Leave room for the main title
plt.show()
