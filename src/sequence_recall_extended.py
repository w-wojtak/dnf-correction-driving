# pylint: disable=C0200
from scipy.ndimage import gaussian_filter1d  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import *
from pathlib import Path
import sys

# ====================================
# -------- Project Setup -------------
# ====================================

PROJECT_ROOT = Path.cwd()
results_dir = PROJECT_ROOT / "results_driving_recall"
results_dir.mkdir(exist_ok=True)

# Trial configuration
trial_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# Driver's actual arrival times for this Monday (in simulation time)
# These represent when the driver actually arrives at each destination
# Trial 1: baseline routine
# Trial 2-3: might have slight variations
actual_arrival_times_by_trial = {
    1: [9, 20, 35, 45],    # coffee, work, gym, home (baseline)
    2: [9, 20, 35, 45],    # same routine
    3: [9, 20, 35, 45],    # same routine
}
actual_arrival_times = actual_arrival_times_by_trial.get(trial_number, 
                                                          actual_arrival_times_by_trial[1])

# ====================================
# -------- Parameters ----------------
# ====================================

# Kernel parameters (same as robotics - they work for any DNF!)
kernel_pars_act = [1.5, 0.8, 0.1]  # action/destination prediction field
kernel_pars_sim = [1.7, 0.8, 0.7]  # simulated partner (human driver)
kernel_pars_wm = [1.75, 0.5, 0.8]  # working memory
kernel_pars_inh = [3, 1.5, 0.0]    # inhibition

# Spatial and temporal parameters
x_lim, t_lim = 80, 60
dx, dt = 0.05, 0.05
x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

# Adaptation and plotting
beta_adapt = 0.001  # learning rate for timing adaptation
plot_fields = False
plot_every = 5
plot_delay = 0.05

# ====================================
# -------- Load Learned Memory -------
# ====================================

folder = PROJECT_ROOT / "results_driving_learning"
file_dest, ts1 = find_latest_file_with_prefix(folder, "u_dest_memory_driverA_monday_")
file_dur, ts2 = find_latest_file_with_prefix(folder, "u_routine_duration_driverA_monday_")
file_meta, ts3 = find_latest_file_with_prefix(folder, "metadata_driverA_monday_")

if not (ts1 == ts2 == ts3):
    raise ValueError("Timestamps do not match across all files.")

u_dest_memory = np.load(file_dest)
u_routine_duration = np.load(file_dur)
metadata = np.load(file_meta, allow_pickle=True).item()

# Extract metadata
destination_positions = metadata['destination_positions']
destination_names = metadata['destination_names']
real_arrival_times = metadata['actual_arrival_times']  # minutes from midnight

print(f"Loaded routine for Driver A, Monday")
print(f"Destinations: {', '.join(destination_names)}")
print(f"Real arrival times: {[f'{t//60:02d}:{t%60:02d}' for t in real_arrival_times]}")

# ====================================
# -------- Input Configuration -------
# ====================================

input_flag = True
input_shape = [3, 1.5]
input_duration = [5, 5, 5, 5]  # how long each arrival event lasts

# System prediction (based on learned memory)
input_onset_time_prediction = [3, 8, 12, 16]  # when system predicts arrivals

# Actual driver arrivals (from GPS or simulation)
input_onset_time_actual = actual_arrival_times

input_pars_prediction = [input_shape, destination_positions, 
                         input_onset_time_prediction, input_duration]
input_pars_actual = [input_shape, destination_positions, 
                     input_onset_time_actual, input_duration]

# Generate input signals
inputs_prediction = get_inputs(x, t, dt, input_pars_prediction, input_flag)
inputs_actual = get_inputs(x, t, dt, input_pars_actual, input_flag)
input_car_feedback = np.zeros((len(t), len(x)))  # car confirms arrivals

# ====================================
# -------- Field Initialization ------
# ====================================

try:
    u_routine_duration = u_routine_duration.flatten()
    h_d_initial = max(u_routine_duration)

    if trial_number == 1:
        # First trial - initialize from learned memory
        u_dest_pred = u_dest_memory - h_d_initial + 1.5
        input_destination_onset_pred = u_dest_memory.flatten()
        h_u_dest_pred = -h_d_initial * np.ones(np.shape(x)) + 1.5

        u_driver_actual = u_dest_memory.flatten() - h_d_initial + 1.5
        input_destination_onset_actual = u_dest_memory.flatten()
        h_u_driver = -h_d_initial * np.ones(np.shape(x)) + 1.5
    else:
        # Load adaptation from previous trial
        print(f"Loading adaptation memory from previous trial...")
        latest_adapt_file, _ = find_latest_file_with_prefix(results_dir, "h_u_adapt_")
        latest_adapt = np.load(latest_adapt_file, allow_pickle=True)

        # Prediction field adjusted by previous adaptations
        u_dest_pred = u_dest_memory.flatten() - h_d_initial + 1.5 - latest_adapt
        input_destination_onset_pred = u_dest_memory.flatten() - latest_adapt
        h_u_dest_pred = -h_d_initial * np.ones(np.shape(x)) + 1.5

        # Driver field (actual behavior) unchanged
        u_driver_actual = u_dest_memory.flatten() - h_d_initial + 1.5
        input_destination_onset_actual = u_dest_memory.flatten()
        h_u_driver = -h_d_initial * np.ones(np.shape(x)) + 1.5

except FileNotFoundError:
    print("No learned memory found, initializing with defaults.")
    u_dest_pred = np.zeros(np.shape(x))
    h_u_dest_pred = -3.2 * np.ones(np.shape(x))

# Working memory parameters
h_0_wm = -1.0
theta_wm = 0.8
u_wm = h_0_wm * np.ones(np.shape(x))
h_u_wm = h_0_wm * np.ones(np.shape(x))

# Thresholds and time constants
tau_h_pred = 20      # prediction field
theta_pred = 1.5
tau_h_driver = 10    # driver actual behavior
theta_driver = 1.5
theta_error = 1.5

# Compute kernels and FFTs
kernel_pred = kernel_gauss(x, *kernel_pars_act)
kernel_driver = kernel_gauss(x, *kernel_pars_sim)
kernel_wm = kernel_osc(x, *kernel_pars_wm)
kernel_inh = kernel_gauss(x, *kernel_pars_inh)

w_hat_pred = np.fft.fft(kernel_pred)
w_hat_driver = np.fft.fft(kernel_driver)
w_hat_wm = np.fft.fft(kernel_wm)
w_hat_inh = np.fft.fft(kernel_inh)

# Feedback/observation fields
h_f = -1.0
w_hat_f = w_hat_pred
tau_h_f = tau_h_pred
theta_f = theta_pred

u_car_obs = h_f * np.ones(np.shape(x))      # car observes predicted arrival
u_driver_obs = h_f * np.ones(np.shape(x))   # car observes actual driver arrival
u_error = h_f * np.ones(np.shape(x))        # prediction error

# Field histories
u_pred_history = []
u_driver_history = []
u_wm_history = []
u_car_obs_history = []
u_driver_obs_history = []

# Adaptation memory field
h_u_adapt = np.zeros(np.shape(x))

# ====================================
# -------- Plotting Setup ------------
# ====================================

fig = axs = None
lines = []

if plot_fields:
    plt.ion()
    fig, axs = plt.subplots(3, 2, figsize=(14, 14), sharex=True)

    fields = [
        (u_dest_pred, 'u_prediction', 0, 0),
        (u_driver_actual, 'u_driver_actual', 0, 1),
        (u_car_obs, 'u_car_observation', 1, 0),
        (u_driver_obs, 'u_driver_observation', 1, 1),
        (u_wm, 'u_working_memory', 2, 0),
    ]
    
    for field, name, row, col in fields:
        line, = axs[row, col].plot(x, field, label=f'{name}(x)')
        axs[row, col].set_ylim(-5, 5)
        axs[row, col].set_ylabel("Activation")
        axs[row, col].legend()
        axs[row, col].set_title(f"{name} - Time = 0")
        
        # Add destination markers
        for dest_pos, dest_name in zip(destination_positions, destination_names):
            axs[row, col].axvline(dest_pos, color='gray', linestyle=':', alpha=0.3)
        
        lines.append(line)
    
    axs[2, 1].axis("off")
    plt.tight_layout()

# ====================================
# -------- Gaussian Feedback Setup ---
# ====================================

destination_indices = [np.argmin(np.abs(x - pos)) for pos in destination_positions]
threshold_crossed = {pos: False for pos in destination_positions}

gaussian_amplitude = 3
gaussian_width = 1.5

def gaussian_input(x, center, amplitude, width):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

# ====================================
# -------- Helper Function -----------
# ====================================

def compute_convolution(u, theta, w_hat):
    """Compute FFT-based convolution for field dynamics."""
    f = np.heaviside(u - theta, 1)
    f_hat = np.fft.fft(f)
    return dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * w_hat)))

# ====================================
# -------- Main Simulation Loop ------
# ====================================

print(f"\nStarting recall simulation (Trial {trial_number})...")
print("System predicts destinations, driver arrives, adaptation occurs.")

for i in range(len(t)):
    input_driver = inputs_actual[i, :]

    # Compute all firing rate functions
    f_car_obs = np.heaviside(u_car_obs - theta_f, 1)
    f_driver_obs = np.heaviside(u_driver_obs - theta_f, 1)
    f_pred = np.heaviside(u_dest_pred - theta_pred, 1)
    f_driver = np.heaviside(u_driver_actual - theta_driver, 1)
    f_wm = np.heaviside(u_wm - theta_wm, 1)
    f_error = np.heaviside(u_error - theta_error, 1)

    # Compute convolutions
    conv_car_obs = compute_convolution(u_car_obs, theta_f, w_hat_f)
    conv_driver_obs = compute_convolution(u_driver_obs, theta_f, w_hat_f)
    conv_pred = compute_convolution(u_dest_pred, theta_pred, w_hat_pred)
    conv_driver = compute_convolution(u_driver_actual, theta_driver, w_hat_driver)
    conv_wm = compute_convolution(u_wm, theta_wm, w_hat_wm)
    conv_inh = dx * np.fft.ifftshift(np.real(np.fft.ifft(np.fft.fft(f_wm) * w_hat_inh)))
    conv_error = compute_convolution(u_error, theta_error, w_hat_pred)

    # Update threshold accommodation
    h_u_dest_pred += dt / tau_h_pred
    h_u_driver += dt / tau_h_driver

    # Update field dynamics
    # Prediction field (system's prediction of when to arrive)
    u_dest_pred += dt * (-u_dest_pred + conv_pred + input_destination_onset_pred + 
                         h_u_dest_pred - 6.0 * f_wm * conv_wm)
    
    # Driver actual behavior field
    u_driver_actual += dt * (-u_driver_actual + conv_driver + input_destination_onset_actual + 
                            h_u_driver - 6.0 * f_wm * conv_wm)
    
    # Working memory (stores executed arrivals)
    u_wm += (dt/1.25) * (-u_wm + conv_wm + 8 * ((f_car_obs * u_car_obs) * 
                         (f_driver_obs * u_driver_obs)) + h_u_wm)
    
    # Observation fields
    u_car_obs += dt * (-u_car_obs + conv_car_obs + input_car_feedback[i, :] + 
                       h_f - 1 * f_wm * conv_wm)
    u_driver_obs += dt * (-u_driver_obs + conv_driver_obs + input_driver + 
                          h_f - 1 * f_wm * conv_wm)
    
    # Error field
    u_error += dt * (-u_error + conv_error + h_f - 2 * f_driver * conv_driver)

    # Adaptation: learn temporal mismatch between prediction and actual arrival
    # If driver arrives before/after prediction, adjust memory
    h_u_adapt += beta_adapt * (1 - (f_driver_obs * f_car_obs)) * (f_car_obs - f_driver_obs)

    # Store history
    u_pred_history.append([u_dest_pred[idx] for idx in destination_indices])
    u_driver_history.append([u_driver_actual[idx] for idx in destination_indices])
    u_wm_history.append([u_wm[idx] for idx in destination_indices])
    u_car_obs_history.append([u_car_obs[idx] for idx in destination_indices])
    u_driver_obs_history.append([u_driver_obs[idx] for idx in destination_indices])

    # Detect when system's prediction crosses threshold → car is ready
    for idx, pos, dest_name in zip(destination_indices, destination_positions, destination_names):
        if not threshold_crossed[pos] and u_dest_pred[idx] > theta_pred:
            print(f"System predicts arrival at {dest_name} (x={pos}) at time {i*dt:.2f}")
            threshold_crossed[pos] = True
            
            # Car sends confirmation feedback after short delay
            time_on = i + 20  # 1 second delay
            time_off = len(t)
            gaussian = gaussian_amplitude * np.exp(-((x - pos) ** 2) / (2 * gaussian_width ** 2))
            input_car_feedback[time_on:time_off, :] += gaussian

    # Update plots
    if plot_fields and (i % plot_every == 0 or i == len(t) - 1):
        lines[0].set_ydata(u_dest_pred)
        lines[1].set_ydata(u_driver_actual)
        lines[2].set_ydata(u_car_obs)
        lines[3].set_ydata(u_driver_obs)
        lines[4].set_ydata(u_wm)

        axs[0, 0].set_title(f"Prediction - Time = {t[i]:.1f}, Trial {trial_number}")
        axs[0, 1].set_title(f"Driver Actual - Time = {t[i]:.1f}")
        axs[1, 0].set_title(f"Car Observation - Time = {t[i]:.1f}")
        axs[1, 1].set_title(f"Driver Observation - Time = {t[i]:.1f}")
        axs[2, 0].set_title(f"Working Memory - Time = {t[i]:.1f}")

        plt.pause(plot_delay)

# ====================================
# -------- Post-Processing -----------
# ====================================

# Smooth adaptation memory
h_u_adapt = gaussian_filter1d(h_u_adapt, sigma=15)

# If not first trial, accumulate with previous adaptation
if trial_number > 1:
    latest_adapt_file, _ = find_latest_file_with_prefix(results_dir, "h_u_adapt_")
    latest_adapt = np.load(latest_adapt_file, allow_pickle=True)
    h_u_adapt = h_u_adapt + latest_adapt

# Save adaptation memory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_adapt = results_dir / f"h_u_adapt_trial{trial_number}_{timestamp}.npy"
np.save(file_adapt, h_u_adapt)
print(f"\nSaved adaptation memory to {file_adapt}")

# ====================================
# -------- Analysis ------------------
# ====================================

# Convert to arrays
u_pred_history = np.array(u_pred_history)
u_driver_history = np.array(u_driver_history)
u_wm_history = np.array(u_wm_history)
u_car_obs_history = np.array(u_car_obs_history)
u_driver_obs_history = np.array(u_driver_obs_history)

timesteps = np.arange(len(u_pred_history)) * dt

# Detect prediction vs actual timing
print("\n" + "="*60)
print(f"Prediction vs. Actual Analysis (Trial {trial_number})")
print("="*60)

for i, (dest_name, dest_pos, real_time) in enumerate(zip(destination_names, 
                                                          destination_positions, 
                                                          real_arrival_times)):
    # When did system predict arrival?
    pred_cross_idx = np.argmax(u_pred_history[:, i] >= theta_pred)
    pred_time = pred_cross_idx * dt
    
    # When did driver actually arrive?
    driver_cross_idx = np.argmax(u_driver_history[:, i] >= theta_driver)
    driver_time = driver_cross_idx * dt
    
    # Timing error
    error = driver_time - pred_time
    
    hours = real_time // 60
    minutes = real_time % 60
    
    print(f"{dest_name:8s} (real {hours:02d}:{minutes:02d}):")
    print(f"  Predicted arrival: t={pred_time:.2f}")
    print(f"  Actual arrival:    t={driver_time:.2f}")
    print(f"  Error: {error:+.2f} {'(late)' if error > 0 else '(early)' if error < 0 else '(perfect)'}")

# ====================================
# -------- Visualization -------------
# ====================================

# Plot 1: Adaptation memory
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, h_u_adapt, label=f'Trial {trial_number} adaptation', linewidth=2)
if trial_number > 1:
    ax.plot(x, latest_adapt, label='Previous trials', linestyle='--', alpha=0.7)

for dest_pos, dest_name in zip(destination_positions, destination_names):
    ax.axvline(dest_pos, color='gray', linestyle=':', alpha=0.5)
    ax.text(dest_pos, ax.get_ylim()[1]*0.9, dest_name, ha='center', fontsize=9)

ax.set_xlabel('Destination space (x)')
ax.set_ylabel('Adaptation value')
ax.set_title(f'Timing Adaptation Memory (Trial {trial_number})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / f"adaptation_trial{trial_number}_{timestamp}.png", dpi=150)
plt.show()

# Plot 2: Field time courses
fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

field_histories = [
    (u_pred_history, 'System Prediction', (-2, 5)),
    (u_driver_history, 'Driver Actual', (-2, 5)),
    (u_wm_history, 'Working Memory', (-2, 25)),
    (u_car_obs_history, 'Car Observation', (-2, 5)),
    (u_driver_obs_history, 'Driver Observation', (-2, 5)),
]

colors_dest = ['tab:brown', 'tab:blue', 'tab:green', 'tab:red']

for ax, (field_hist, name, ylim) in zip(axs, field_histories):
    for pos_idx, (dest_name, color) in enumerate(zip(destination_names, colors_dest)):
        ax.plot(timesteps, field_hist[:, pos_idx], 
                label=f'{dest_name} (x={destination_positions[pos_idx]})',
                color=color, linewidth=1.5)
    ax.axhline(theta_pred, color='k', linestyle='--', linewidth=0.8, alpha=0.5, label='threshold')
    ax.set_ylabel(name)
    ax.set_ylim(ylim)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axs[-1].set_xlabel('Simulation Time')
fig.suptitle(f'Destination Recall - Trial {trial_number} - Driver A Monday', fontsize=14)
plt.tight_layout()
plt.savefig(results_dir / f"timecourses_trial{trial_number}_{timestamp}.png", dpi=150)
plt.show()

# Plot 3: Prediction error over time
fig, ax = plt.subplots(figsize=(10, 5))

for pos_idx, (dest_name, color) in enumerate(zip(destination_names, colors_dest)):
    error_signal = u_driver_history[:, pos_idx] - u_pred_history[:, pos_idx]
    ax.plot(timesteps, error_signal, label=dest_name, color=color, linewidth=2)

ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Simulation Time')
ax.set_ylabel('Prediction Error (driver - prediction)')
ax.set_title(f'Timing Prediction Error (Trial {trial_number})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / f"error_trial{trial_number}_{timestamp}.png", dpi=150)
plt.show()

print("\n" + "="*60)
print("Recall simulation complete!")
print("="*60)