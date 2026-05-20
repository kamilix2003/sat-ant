import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# --- Physical Constants ---
c = 3e8         # Speed of light (m/s)
fc = 2.2e9      # Carrier frequency (2.2 GHz S-Band, common for space telemetry)

# --- Ground Station Setup (3D) ---
# 5 stations spread across a 2000 km x 2000 km continental area at Z=0 (sea level)
sensors = np.array([
    [0, 0, 0],                     # Reference Station
    [2000e3, 0, 0],                # East
    [0, 2000e3, 0],                # North
    [2000e3, 2000e3, 0],           # North-East
    [1000e3, 1000e3, 0]            # Center
])

def compute_measurements(state, sensors):
    """
    Computes ideal TDOA and FDOA measurements for a 3D target state.
    state: [x, y, z, vx, vy, vz]
    """
    pos = state[:3]
    vel = state[3:]
    
    # 3D Range to each sensor
    vectors = pos - sensors
    ranges = np.linalg.norm(vectors, axis=1)
    
    # Radial velocity (range rate) to each sensor
    # r_dot = (vector_to_target dot velocity) / range
    range_rates = np.sum(vectors * vel, axis=1) / ranges
    
    # TDOA in seconds (relative to Sensor 0)
    tdoa = (ranges[1:] - ranges[0]) / c
    
    # FDOA in Hz (Doppler shift difference relative to Sensor 0)
    fdoa = -(range_rates[1:] - range_rates[0]) * (fc / c)
    
    return np.concatenate([tdoa, fdoa])

def objective_function(state, sensors, meas_noisy, sigma_tdoa, sigma_fdoa):
    """Residuals function for least_squares optimizer."""
    meas_model = compute_measurements(state, sensors)
    
    # Weight residuals by inverse variance
    res_tdoa = (meas_model[:4] - meas_noisy[:4]) / sigma_tdoa
    res_fdoa = (meas_model[4:] - meas_noisy[4:]) / sigma_fdoa
    
    return np.concatenate([res_tdoa, res_fdoa])

def run_monte_carlo(true_state, sensors, num_trials=200):
    """Runs Monte Carlo simulation for 3D LEO tracking."""
    # Realistic noise for satellite telemetry
    sigma_tdoa = 500e-9  # 50 nanoseconds timing error
    sigma_fdoa = 50.0    # 5 Hz frequency error
    
    true_meas = compute_measurements(true_state, sensors)
    est_positions = []
    
    for _ in range(num_trials):
        noise_tdoa = np.random.randn(4) * sigma_tdoa
        noise_fdoa = np.random.randn(4) * sigma_fdoa
        meas_noisy = true_meas + np.concatenate([noise_tdoa, noise_fdoa])
        
        # Initial guess (simulating radar handoff: 20km pos error, 50m/s vel error)
        guess = true_state + np.array([20000, -20000, 20000, 50, -50, 10])
        
        # Solve using Levenberg-Marquardt
        res = least_squares(
            objective_function, 
            guess, 
            args=(sensors, meas_noisy, sigma_tdoa, sigma_fdoa),
            method='lm'
        )
        
        if res.success:
            est_positions.append(res.x[:3]) # Capture x, y, z
            
    return np.array(est_positions)

# --- Run Scenarios ---
np.random.seed(42)

# Scenario 1: Spacecraft directly over the array
# Position: Center of array, 500km altitude. Velocity: 7.5 km/s primarily in X.
state_zenith = np.array([1000e3, 1000e3, 500e3, 7500, 1000, 0]) 
scatter_zenith = run_monte_carlo(state_zenith, sensors)

# Scenario 2: Spacecraft near the horizon
# Position: 4000km away from array center.
state_horizon = np.array([5000e3, 5000e3, 500e3, 7500, 1000, 0])
scatter_horizon = run_monte_carlo(state_horizon, sensors)

# --- 3D Visualization ---
fig = plt.figure(figsize=(16, 7))

# Plot Good Geometry (Zenith)
ax1 = fig.add_subplot(121, projection='3d')
# Convert meters to kilometers for plotting readability
sens_km = sensors / 1000
scat_zen_km = scatter_zenith / 1000
true_zen_km = state_zenith[:3] / 1000

ax1.scatter(sens_km[:, 0], sens_km[:, 1], sens_km[:, 2], c='black', marker='^', s=100, label='Ground Stations')
ax1.scatter(scat_zen_km[:, 0], scat_zen_km[:, 1], scat_zen_km[:, 2], c='blue', alpha=0.3, s=10, label='Est. Positions')
ax1.scatter(true_zen_km[0], true_zen_km[1], true_zen_km[2], c='red', marker='x', s=100, label='True Target')
ax1.set_title("Overhead Pass (Zenith)\nModerate Z-Axis Sensitivity")
ax1.set_xlabel("X (km)")
ax1.set_ylabel("Y (km)")
ax1.set_zlabel("Altitude Z (km)")
ax1.legend()

# Plot Poor Geometry (Horizon)
ax2 = fig.add_subplot(122, projection='3d')
scat_hor_km = scatter_horizon / 1000
true_hor_km = state_horizon[:3] / 1000

ax2.scatter(sens_km[:, 0], sens_km[:, 1], sens_km[:, 2], c='black', marker='^', s=100, label='Ground Stations')
ax2.scatter(scat_hor_km[:, 0], scat_hor_km[:, 1], scat_hor_km[:, 2], c='blue', alpha=0.3, s=10, label='Est. Positions')
ax2.scatter(true_hor_km[0], true_hor_km[1], true_hor_km[2], c='red', marker='x', s=100, label='True Target')
ax2.set_title("Low Elevation Pass (Horizon)\nSevere Line-of-Sight Sensitivity")
ax2.set_xlabel("X (km)")
ax2.set_ylabel("Y (km)")
ax2.set_zlabel("Altitude Z (km)")
ax2.legend()

plt.tight_layout()
plt.show()