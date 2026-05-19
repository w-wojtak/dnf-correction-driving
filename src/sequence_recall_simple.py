# pylint: disable=C0200
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import *
from pathlib import Path

# ====================================
# -------- Project Setup -------------
# ====================================

PROJECT_ROOT = Path.cwd()
results_dir = PROJECT_ROOT / "results_driving_recall"
results_dir.mkdir(exist_ok=True)

# ====================================
# -------- Parameters ----------------
# ====================================

# Kernel parameters
kernel_pars_decision = [1.5, 0.8, 0.1]  # decision field for triggering predictions
kernel_pars_wm = [1.75, 0.5, 0.8]       # working memory

# Spatial and temporal parameters
x_lim, t_lim = 80, 60
dx, dt = 0.05, 0.05
x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

# Plotting
plot_fields = True
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
real_arrival_times = metadata['actual_arrival_times']

print(f"\nLoaded routine for Driver A, Monday")
print(f"Destinations: {', '.join(destination_names)}")
print(f"Real arrival times: {[f'{t//60:02d}:{t%60:02d}' for t in real_arrival_times]}\n")

# ====================================
# -------- Field Initialization ------
# ====================================

u_routine_duration = u_routine_duration.flatten()
h_d_initial = max(u_routine_duration)

# Initialize decision field from learned memory
u_decision = u_dest_memory - h_d_initial + 1.5
h_u_decision = -h_d_initial * np.ones(len(x)) + 1.5

# Working memory (stores which destinations have been recalled)
h_0_wm = -1.0
theta_wm = 0.8
u_wm = h_0_wm * np.ones(len(x))
h_u_wm = h_0_wm * np.ones(len(x))

# Thresholds and time constants
tau_h_decision = 20
theta_decision = 1.5
tau_h_duration = 20

# Compute kernels and FFTs
kernel_decision = kernel_gauss(x, *kernel_pars_decision)
kernel_wm = kernel_osc(x, *kernel_pars_wm)

w_hat_decision = np.fft.fft(kernel_decision)
w_hat_wm = np.fft.fft(kernel_wm)

# Field histories
u_decision_history = []
u_wm_history = []

# Find destination indices
destination_indices = [np.argmin(np.abs(x - pos)) for pos in destination_positions]

# ====================================
# -------- Plotting Setup ------------
# ====================================

fig = axs = line_decision = line_wm = None

if plot_fields:
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Decision field
    line_decision, = axs[0].plot(x, u_decision, label='Decision field u_decision(x)')
    axs[0].set_ylim(-5, 5)
    axs[0].set_ylabel("Activation")
    axs[0].legend()
    axs[0].set_title("Decision Field - Time = 0")
    axs[0].grid(True, alpha=0.3)
    
    # Add destination markers
    for dest_pos, dest_name in zip(destination_positions, destination_names):
        axs[0].axvline(dest_pos, color='gray', linestyle=':', alpha=0.3)
        axs[0].text(dest_pos, axs[0].get_ylim()[1]*0.9, dest_name, 
                   ha='center', fontsize=9)

    # Working memory field
    line_wm, = axs[1].plot(x, u_wm, label='Working memory u_wm(x)')
    axs[1].set_ylim(-5, 5)
    axs[1].set_xlabel("Destination space (x)")
    axs[1].set_ylabel("Activation")
    axs[1].legend()
    axs[1].set_title("Working Memory - Time = 0")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()

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

print("Recalling and executing learned routine...\n")

threshold_crossed = {pos: False for pos in destination_positions}
predicted_destinations = []

for i in range(len(t)):
    # Compute firing rates
    f_decision = np.heaviside(u_decision - theta_decision, 1)
    f_wm = np.heaviside(u_wm - theta_wm, 1)

    # Compute convolutions
    conv_decision = compute_convolution(u_decision, theta_decision, w_hat_decision)
    conv_wm = compute_convolution(u_wm, theta_wm, w_hat_wm)

    # Update threshold accommodation
    h_u_decision += dt / tau_h_decision

    # Update field dynamics
    # Decision field: predicts next destination based on memory
    u_decision += dt * (-u_decision + conv_decision + u_dest_memory + 
                        h_u_decision - 6.0 * f_wm * conv_wm)
    
    # Working memory: stores recalled destinations
    u_wm += (dt/1.25) * (-u_wm + conv_wm + 8.0 * (f_decision * u_decision) + h_u_wm)

    # Store history
    u_decision_history.append([u_decision[idx] for idx in destination_indices])
    u_wm_history.append([u_wm[idx] for idx in destination_indices])

    # Check threshold crossings to detect predicted arrivals
    for idx, pos, dest_name in zip(destination_indices, destination_positions, destination_names):
        if not threshold_crossed[pos] and u_decision[idx] > theta_decision:
            arrival_time_min = real_arrival_times[destination_names.index(dest_name)]
            hours = arrival_time_min // 60
            minutes = arrival_time_min % 60
            print(f"Predicted arrival: {dest_name:8s} at {hours:02d}:{minutes:02d} "
                  f"(sim time = {i*dt:.2f})")
            threshold_crossed[pos] = True
            predicted_destinations.append((dest_name, arrival_time_min))

    # Update plots
    if plot_fields and (i % plot_every == 0 or i == len(t) - 1):
        line_decision.set_ydata(u_decision)
        line_wm.set_ydata(u_wm)
        
        axs[0].set_title(f"Decision Field - Sim Time = {t[i]:.2f}")
        axs[1].set_title(f"Working Memory - Sim Time = {t[i]:.2f}")
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(plot_delay)

plt.ioff()
if plot_fields:
    plt.show()

# ====================================
# -------- Analysis ------------------
# ====================================

u_decision_history = np.array(u_decision_history)
u_wm_history = np.array(u_wm_history)
timesteps = np.arange(len(u_decision_history)) * dt

print("\n" + "="*60)
print("Recall Summary")
print("="*60)
print(f"Total destinations predicted: {len(predicted_destinations)}")
for dest_name, arrival_time in predicted_destinations:
    hours = arrival_time // 60
    minutes = arrival_time % 60
    print(f"  - {dest_name} ({hours:02d}:{minutes:02d})")

print("="*60 + "\n")

# ====================================
# -------- Visualization -------------
# ====================================

# Plot 1: Decision and working memory time courses
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

colors_dest = ['tab:brown', 'tab:blue', 'tab:green', 'tab:red']

# Decision field
for i, (dest_name, color) in enumerate(zip(destination_names, colors_dest)):
    axs[0].plot(timesteps, u_decision_history[:, i], 
                label=dest_name, color=color, linewidth=2)
axs[0].axhline(theta_decision, color='k', linestyle='--', 
               linewidth=1, alpha=0.5, label='threshold')
axs[0].set_ylabel('Decision Field Activation')
axs[0].set_title('Decision Field - Predicting Destinations Over Time')
axs[0].set_ylim(0, 2)
axs[0].legend(loc='upper left')
axs[0].grid(True, alpha=0.3)

# Working memory
for i, (dest_name, color) in enumerate(zip(destination_names, colors_dest)):
    axs[1].plot(timesteps, u_wm_history[:, i], 
                label=dest_name, color=color, linewidth=2)
axs[1].axhline(theta_wm, color='k', linestyle='--', 
               linewidth=1, alpha=0.5, label='threshold')
axs[1].set_ylabel('Working Memory Activation')
axs[1].set_xlabel('Simulation Time')
axs[1].set_title('Working Memory - Storing Recalled Destinations')
axs[1].set_ylim(-2, 4)
axs[1].legend(loc='upper left')
axs[1].grid(True, alpha=0.3)

plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(results_dir / f"recall_analysis_{timestamp}.png", dpi=150)
plt.show()

# Plot 2: Final spatial profile of decision field
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, u_decision, color='black', linewidth=2, alpha=0.3, label='Full field')

for i, (dest_pos, dest_name, color) in enumerate(zip(destination_positions, 
                                                       destination_names, 
                                                       colors_dest)):
    window = 50
    idx = destination_indices[i]
    start_idx = max(0, idx - window)
    end_idx = min(len(x), idx + window)
    
    amplitude = u_decision[idx]
    arrival_time = real_arrival_times[i]
    hours = arrival_time // 60
    minutes = arrival_time % 60
    
    ax.plot(x[start_idx:end_idx], u_decision[start_idx:end_idx], 
            color=color, linewidth=3, 
            label=f'{dest_name} ({hours:02d}:{minutes:02d}) - amp: {amplitude:.2f}')
    ax.axvline(dest_pos, color=color, linestyle=':', alpha=0.5)
    ax.scatter([dest_pos], [amplitude], color=color, s=100, zorder=5)

ax.axhline(theta_decision, color='k', linestyle='--', 
           label='threshold', linewidth=1)
ax.set_xlabel('Destination Space (x)')
ax.set_ylabel('Decision Field Activation')
ax.set_title('Final Decision Field Profile (Ready to Recall)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / f"decision_field_profile_{timestamp}.png", dpi=150)
plt.show()

print(f"Results saved to {results_dir}")