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
results_dir = PROJECT_ROOT / "results_driving_learning"
results_dir.mkdir(exist_ok=True)

# ====================================
# -------- Parameters ----------------
# ====================================

# Kernel and spatial/temporal parameters
kernel_pars = [1, 0.7, 0.9]
x_lim, t_lim = 80, 60  # x represents destination space (like before)
dx, dt = 0.05, 0.05
theta = 1

# Field parameters
tau_h = 20
h_0 = 0
h_0_d = 0
tau_h_d = 20

# Input configuration
# X-axis: destination positions (arbitrary spatial encoding)
# Time: when destinations are visited (encoded in peak amplitude via onset timing)
input_flag = True
input_shape = [3, 1.5]
input_duration = [1, 1, 1, 1]

# Destination positions in abstract "destination space"
destination_positions = [-60, -30, 0, 30]  # coffee, work, gym, home
destination_names = ["coffee", "work", "gym", "home"]
destination_colors = ["tab:brown", "tab:blue", "tab:green", "tab:red"]

# Actual arrival times (minutes from midnight) - used ONLY to determine onset order
# coffee: 8:00 = 480 min
# work:   9:00 = 540 min
# gym:   18:00 = 1080 min
# home:  20:00 = 1200 min
actual_arrival_times = [480, 540, 1080, 1200]

# Convert to onset times in simulation (earlier visit = earlier onset = higher amplitude)
# Normalize to simulation time scale (0-60)
min_time = min(actual_arrival_times)
max_time = max(actual_arrival_times)
time_range = max_time - min_time

# Map actual times to simulation onset times
# Earlier arrivals get earlier onsets → higher final amplitude
input_onset_time = [
    9 + (time - min_time) / time_range * 35  # maps to range [9, 44]
    for time in actual_arrival_times
]

input_pars = [input_shape, destination_positions, input_onset_time, input_duration]

# Plotting parameters
plot_fields = True
plot_every = 5
plot_delay = 0.05

# ====================================
# -------- Initialization ------------
# ====================================

x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

# Find indices for each destination position
destination_indices = [np.argmin(np.abs(x - pos)) for pos in destination_positions]

# Generate input sequence (simulates observing driver's routine)
inputs = get_inputs(x, t, dt, input_pars, input_flag)

# Initialize fields
u_dest_memory = h_0 * np.ones_like(x)
h_u_dest = h_0 * np.ones_like(x)
u_routine_duration = h_0_d * np.ones_like(x)
h_u_duration = h_0_d * np.ones_like(x)

# Precompute kernel FFT
w_hat = np.fft.fft(kernel_osc(x, *kernel_pars))

# History storage
u_dest_tc = []
u_dur_tc = []

# ====================================
# -------- Plot Setup ----------------
# ====================================

fig = axs = line1_field = line1_input = line2_field = None

if plot_fields:
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Destination sequence memory field
    line1_field, = axs[0].plot(x, u_dest_memory, label='Destination memory u_dest(x)')
    line1_input, = axs[0].plot(x, inputs[0, :], label='GPS Input', linestyle='--')
    axs[0].set_ylim(-2, 10)
    axs[0].set_ylabel("Activation")
    axs[0].set_xlabel("Destination space (x)")
    axs[0].legend()
    axs[0].set_title("Destination Sequence Memory - Time = 0")
    axs[0].grid(True, alpha=0.3)

    # Add vertical lines and labels for destinations
    for dest_pos, dest_name, color in zip(destination_positions, destination_names, destination_colors):
        axs[0].axvline(dest_pos, color=color, linestyle=':', alpha=0.5, linewidth=1)
        axs[0].text(dest_pos, axs[0].get_ylim()[1]*0.95, dest_name, 
                   ha='center', fontsize=9, color=color)

    # Routine duration field
    line2_field, = axs[1].plot(x, u_routine_duration, label='Routine duration u_dur(x)')
    axs[1].set_ylim(-2, 10)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Activation")
    axs[1].legend()
    axs[1].set_title("Routine Duration Field - Time = 0")
    axs[1].grid(True, alpha=0.3)

# ====================================
# -------- Helper Function -----------
# ====================================

def compute_field_update(u, theta, w_hat, h_u, tau_h, input_signal):
    """Compute field dynamics update with threshold accommodation."""
    f = np.heaviside(u - theta, 1)
    f_hat = np.fft.fft(f)
    conv = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * w_hat)))
    
    h_u_new = h_u + dt / tau_h * f  # threshold accommodation
    u_new = u + dt * (-u + conv + input_signal + h_u_new)
    
    return u_new, h_u_new

# ====================================
# -------- Simulation Loop -----------
# ====================================

print("Learning Driver A's Monday routine...")
print(f"Destinations: {', '.join(destination_names)}")
print(f"Arrival times: {[f'{t//60:02d}:{t%60:02d}' for t in actual_arrival_times]}")
print(f"Input onsets: {[f'{t:.1f}' for t in input_onset_time]}\n")

for i in range(len(t)):
    # Routine duration field input (triggered at beginning)
    input_duration_signal = 3.0 * np.exp(-((x - 0) ** 2) / (2 * 1.5 ** 2)) if i < 1/dt else 0.0

    # Update destination sequence memory field
    u_dest_memory, h_u_dest = compute_field_update(
        u_dest_memory, theta, w_hat, h_u_dest, tau_h, inputs[i, :]
    )
    
    # Update routine duration field
    u_routine_duration, h_u_duration = compute_field_update(
        u_routine_duration, theta, w_hat, h_u_duration, tau_h_d, input_duration_signal
    )

    # Store time course data at destination locations
    u_dest_tc.append([u_dest_memory[idx] for idx in destination_indices])
    u_dur_tc.append(u_routine_duration[int(len(x) / 2)])

    # Update plots
    if plot_fields and (i % plot_every == 0 or i == len(t) - 1):
        line1_field.set_ydata(u_dest_memory)
        line1_input.set_ydata(inputs[i, :])
        line2_field.set_ydata(u_routine_duration)
        
        axs[0].set_title(f"Destination Sequence Memory - Simulation Time = {t[i]:.2f}")
        axs[1].set_title(f"Routine Duration Field - Simulation Time = {t[i]:.2f}")
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(plot_delay)

print(f"\nLearning complete!")
print(f"Max activation in destination memory: {max(u_dest_memory):.2f}")
print(f"Max activation in duration field: {max(u_routine_duration):.2f}")

# ====================================
# -------- Save Results --------------
# ====================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_dest_memory = results_dir / f"u_dest_memory_driverA_monday_{timestamp}.npy"
file_duration = results_dir / f"u_routine_duration_driverA_monday_{timestamp}.npy"

# Also save metadata
metadata = {
    'destination_positions': destination_positions,
    'destination_names': destination_names,
    'actual_arrival_times': actual_arrival_times,
    'input_onset_time': input_onset_time
}
file_metadata = results_dir / f"metadata_driverA_monday_{timestamp}.npy"
np.save(file_metadata, metadata)

np.save(file_dest_memory, u_dest_memory)
np.save(file_duration, u_routine_duration)

print(f"\nSaved destination memory to {file_dest_memory}")
print(f"Saved routine duration to {file_duration}")
print(f"Saved metadata to {file_metadata}")

plt.ioff()
if plot_fields:
    plt.show()

# ====================================
# -------- Analysis ------------------
# ====================================

u_dest_history = np.array(u_dest_tc)
u_dur_history = np.array(u_dur_tc)
timesteps = np.arange(len(u_dest_history))

# Find threshold crossings
print("\n" + "="*50)
print("Threshold Crossing Analysis")
print("="*50)
print("(Earlier crossing = earlier in route = higher final amplitude)\n")

for i, (dest_name, actual_time) in enumerate(zip(destination_names, actual_arrival_times)):
    crossing_idx = np.argmax(u_dest_history[:, i] >= theta)
    final_amplitude = u_dest_history[-1, i]
    hours = actual_time // 60
    minutes = actual_time % 60
    print(f"{dest_name:8s} (real arrival {hours:02d}:{minutes:02d}) - "
          f"crosses threshold at sim time {crossing_idx*dt:.2f} - "
          f"final amplitude: {final_amplitude:.2f}")

crossing_idx_dur = np.argmax(u_dur_history >= theta)
print(f"\nDuration field crosses threshold at simulation time {crossing_idx_dur*dt:.2f}")

# ====================================
# -------- Visualization -------------
# ====================================

fig, axs = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Destination memory time courses
for i, (dest_name, color) in enumerate(zip(destination_names, destination_colors)):
    axs[0].plot(timesteps * dt, u_dest_history[:, i], 
                label=f'{dest_name} ({actual_arrival_times[i]//60:02d}:{actual_arrival_times[i]%60:02d})', 
                color=color, linewidth=2)
axs[0].axhline(theta, color='k', linestyle='--', label='threshold', linewidth=1)
axs[0].set_ylabel('Memory Activation')
axs[0].set_xlabel('Simulation Time')
axs[0].set_title('Destination Memory Evolution (Earlier visits reach higher amplitude)')
axs[0].set_ylim(-1, 5)
axs[0].legend(loc='upper left')
axs[0].grid(True, alpha=0.3)

# Plot 2: Routine duration time course
axs[1].plot(timesteps * dt, u_dur_history, label='Routine duration', 
            color='tab:purple', linewidth=2)
axs[1].axhline(theta, color='k', linestyle='--', label='threshold', linewidth=1)
axs[1].set_ylabel('Duration Field Activation')
axs[1].set_xlabel('Simulation Time')
axs[1].set_title('Routine Duration Field (24h timer)')
axs[1].set_ylim(-1, 3)
axs[1].set_xlim(0, 100*dt)
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Plot 3: Final spatial profile of destination memory
axs[2].plot(x, u_dest_memory, color='black', linewidth=2, alpha=0.3, label='Full field')

for i, (dest_pos, dest_name, color) in enumerate(zip(destination_positions, 
                                                       destination_names, 
                                                       destination_colors)):
    # Highlight individual peaks
    window = 50
    idx = destination_indices[i]
    start_idx = max(0, idx - window)
    end_idx = min(len(x), idx + window)
    
    amplitude = u_dest_memory[idx]
    actual_time = actual_arrival_times[i]
    
    axs[2].plot(x[start_idx:end_idx], u_dest_memory[start_idx:end_idx], 
                color=color, linewidth=3, 
                label=f'{dest_name} ({actual_time//60:02d}:{actual_time%60:02d}) - amp: {amplitude:.2f}')
    axs[2].axvline(dest_pos, color=color, linestyle=':', alpha=0.5)
    axs[2].scatter([dest_pos], [amplitude], color=color, s=100, zorder=5)

axs[2].axhline(theta, color='k', linestyle='--', label='threshold', linewidth=1)
axs[2].set_xlabel('Destination Space (x) - each position = one destination')
axs[2].set_ylabel('Memory Activation')
axs[2].set_title('Final Destination Memory Profile (Peak amplitude = temporal order)')
axs[2].legend(loc='upper right', fontsize=9)
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / f"learning_analysis_{timestamp}.png", dpi=150)
plt.show()

# ====================================
# -------- Summary Statistics --------
# ====================================

print("\n" + "="*50)
print("Learning Summary")
print("="*50)
print(f"Driver: A")
print(f"Day: Monday")
print(f"Number of destinations: {len(destination_names)}")
print(f"\nDestination sequence (by peak amplitude = temporal order):")

# Get peak amplitudes and sort
peak_amplitudes = [u_dest_memory[idx] for idx in destination_indices]
sorted_indices = np.argsort(peak_amplitudes)[::-1]  # descending

for rank, idx in enumerate(sorted_indices, 1):
    dest_name = destination_names[idx]
    dest_pos = destination_positions[idx]
    amplitude = peak_amplitudes[idx]
    actual_time = actual_arrival_times[idx]
    hours = actual_time // 60
    minutes = actual_time % 60
    print(f"  {rank}. {dest_name:8s} at x={dest_pos:+4.0f} - "
          f"real time {hours:02d}:{minutes:02d} - amplitude: {amplitude:.2f}")

print(f"\nRoutine duration: {max(u_routine_duration):.2f}")
print("="*50)

print("\nKey insight:")
print("- X-axis position = WHICH destination (coffee, work, gym, home)")
print("- Peak amplitude = WHEN visited (higher = earlier in the day)")
print("- This is exactly like robotics: x=which object, amplitude=when to grasp")