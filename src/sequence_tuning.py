# pylint: disable=C0200

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from dataclasses import dataclass
from enum import Enum


# ====================================
# -------- Human feedback ------------
# ====================================

class FeedbackType(Enum):
    SKIP  = "skip"
    LOCK  = "lock"
    EARLY = "early"
    LATE  = "late"
    SWAP  = "swap"
    # TODO: ADD = "add"


@dataclass
class FeedbackFieldParams:
    kernel_type: str   # "gauss" or "osc"
    kernel_pars: list
    h_0: float         # resting level
    theta: float       # threshold
    tau: float         # time constant
    transient: bool    # True = decaying, False = sustained


class FeedbackField:
    def __init__(self, x, dx, params: FeedbackFieldParams, input_duration):
        self.x = x
        self.dx = dx
        self.params = params
        self.u = params.h_0 * np.ones(len(x))
        self.h = params.h_0 * np.ones(len(x))
        self.s = np.zeros(len(x))
        self.input_duration = input_duration
        self._inject_step = None

        if params.kernel_type == "gauss":
            self.kernel = kernel_gauss(x, *params.kernel_pars)
        elif params.kernel_type == "osc":
            self.kernel = kernel_osc(x, *params.kernel_pars)
        self.w_hat = np.fft.fft(self.kernel)

    def inject(self, center, amplitude, width, current_step):
        """Set Gaussian external input — applied every step until cleared."""
        self.s = amplitude * np.exp(-0.5 * ((self.x - center) / width) ** 2)
        self._inject_step = current_step

    def inject_add(self, center, amplitude, width):
        """Add a second Gaussian to existing input — used for SWAP."""
        self.s += amplitude * np.exp(-0.5 * ((self.x - center) / width) ** 2)

    def clear_input(self):
        self.s = np.zeros(len(self.x))
        self._inject_step = None

    def output(self):
        return np.heaviside(self.u - self.params.theta, 1.0)


# ====================================
# -------- Initialization ------------
# ====================================

x_lim, t_lim = 80, 60
dx, dt = 0.05, 0.05

x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)


# ====================================
# -------- Project paths -------------
# ====================================

PROJECT_ROOT = Path.cwd()
results_dir  = PROJECT_ROOT / "results_driving_correction"
results_dir.mkdir(exist_ok=True)


# ====================================
# -------- Load memory ---------------
# ====================================

folder = PROJECT_ROOT / "results_driving_learning"
file_dest, ts1 = find_latest_file_with_prefix(folder, "u_dest_memory_driverA_monday_")
file_dur, ts2 = find_latest_file_with_prefix(folder, "u_routine_duration_driverA_monday_")
file_meta, ts3 = find_latest_file_with_prefix(folder, "metadata_driverA_monday_")

if not (ts1 == ts2 == ts3):
    raise ValueError("Timestamps do not match across all files.")

u_dest_memory_loaded = np.load(file_dest)
u_routine_duration = np.load(file_dur)
metadata = np.load(file_meta, allow_pickle=True).item()

# Extract metadata
destination_positions = metadata['destination_positions']
destination_names = metadata['destination_names']
real_arrival_times = metadata['actual_arrival_times']
destination_colors = ['tab:brown', 'tab:blue', 'tab:green', 'tab:red']

print(f"Loaded routine for Driver A, Monday")
print(f"Destinations: {', '.join(destination_names)}")
print(f"Positions: {destination_positions}")
print(f"Real arrival times: {[f'{t//60:02d}:{t%60:02d}' for t in real_arrival_times]}")

# Compute destination index positions
destination_indices = [np.argmin(np.abs(x - pos)) for pos in destination_positions]


def compute_destination_bounds(x, positions):
    """Compute spatial index ranges for each destination bucket."""
    positions = np.array(positions)
    midpoints = (positions[:-1] + positions[1:]) / 2
    boundaries = np.concatenate(([x[0]], midpoints, [x[-1]]))
    return [np.where((x >= boundaries[i]) & (x < boundaries[i + 1]))[0]
            for i in range(len(positions))]

destination_buckets = compute_destination_bounds(x, destination_positions)


# ====================================
# -------- Field initialization ------
# ====================================

try:
    u_routine_duration = u_routine_duration.flatten()
    h_d_initial = max(u_routine_duration)
    input_destination_onset = u_dest_memory_loaded.flatten()
    u_dest_memory = u_dest_memory_loaded.copy()

except FileNotFoundError:
    print("No previous sequence memory found, initializing with default values.")
    h_d_initial = 3.2
    input_destination_onset = np.zeros(len(x))
    u_dest_memory = np.zeros(len(x))


# ====================================
# -------- Field parameters ----------
# ====================================

kernel_pars_dest = [1.5, 0.8, 0.1]
kernel_pars_wm = [1.75, 0.5, 0.8]

h_0_wm = -1.0
theta_wm = 0.8
tau_h_dest = 20
theta_dest = 1.5


# ====================================
# -------- Kernels & FFTs ------------
# ====================================

kernel_dest = kernel_gauss(x, *kernel_pars_dest)
kernel_wm = kernel_osc(x, *kernel_pars_wm)

w_hat_dest = np.fft.fft(kernel_dest)
w_hat_wm = np.fft.fft(kernel_wm)


# ====================================
# -------- Helper functions ----------
# ====================================

def compute_convolution(u, theta, w_hat):
    """Compute FFT-based convolution for field dynamics."""
    f = np.heaviside(u - theta, 1)
    f_hat = np.fft.fft(f)
    return dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * w_hat)))


def reset_execution_fields(u_dest_memory, h_d_initial, x):
    """Re-initialize execution fields from current memory."""
    u_dest = u_dest_memory.copy() - h_d_initial + 1.5
    u_wm = h_0_wm * np.ones(len(x))
    h_u_dest = -h_d_initial * np.ones(len(x)) + 1.5
    return u_dest, u_wm, h_u_dest


def resolve_feedback(destination_name, destination_names, destination_positions):
    """Map destination name to spatial center."""
    idx = [n.lower() for n in destination_names].index(destination_name.lower())
    return destination_positions[idx]


def reset_feedback_fields(feedback_fields):
    """Bring all feedback fields back to resting state."""
    for ff in feedback_fields.values():
        ff.u = ff.params.h_0 * np.ones(len(ff.x))
        ff.clear_input()


def apply_skip_from_field(u_dest_memory, skip_field):
    """Flatten memory region where skip field is active."""
    u_new = u_dest_memory.copy()
    u_new[skip_field.output() > 0] = u_dest_memory[0]
    return u_new


def apply_lock_from_field(h_dmem_mask, lock_field):
    """Zero out mask where lock field is active."""
    mask_new = h_dmem_mask.copy()
    mask_new[lock_field.output() > 0] = 0.0
    return mask_new


early_late_coupling = 0.5  # small enough to keep peak above threshold

def apply_early_from_field(u_dest_memory, early_field, theta_dest):
    """Weaken memory peak where early field is active → recalls earlier."""
    u_new = u_dest_memory.copy()
    u_new -= early_late_coupling * early_field.output() * np.heaviside(u_dest_memory - theta_dest, 1.0)
    return u_new


def apply_late_from_field(u_dest_memory, late_field, theta_dest):
    """Strengthen memory peak where late field is active → recalls later."""
    u_new = u_dest_memory.copy()
    u_new += early_late_coupling * late_field.output() * np.heaviside(u_dest_memory - theta_dest, 1.0)
    return u_new


def apply_swap_from_field(u_dest_memory, swap_field):
    """Swap memory values between two active regions of swap field."""
    u_new = u_dest_memory.copy()
    active = swap_field.output() > 0

    # find two separate 1D regions in the binary output
    regions = []
    in_region = False
    start = 0
    for i in range(len(active)):
        if active[i] and not in_region:
            start = i
            in_region = True
        elif not active[i] and in_region:
            regions.append(np.arange(start, i))
            in_region = False
    if in_region:
        regions.append(np.arange(start, len(active)))

    if len(regions) != 2:
        print(f"[WARNING] SWAP expected 2 active regions, found {len(regions)}. Skipping.")
        return u_dest_memory

    idx_a, idx_b = regions[0], regions[1]
    u_new[idx_a], u_new[idx_b] = u_new[idx_b].copy(), u_new[idx_a].copy()
    return u_new


# ====================================
# -------- Feedback field setup ------
# ====================================

input_duration = 50  # steps, same for all feedback fields

feedback_field_params = FeedbackFieldParams(
    kernel_type="gauss",
    kernel_pars=[2.0, 0.8, 0.05],
    h_0=-1.0,
    theta=0.5,
    tau=1.0,
    transient=False
)

feedback_fields = {
    FeedbackType.SKIP:  FeedbackField(x, dx, feedback_field_params, input_duration),
    FeedbackType.LOCK:  FeedbackField(x, dx, feedback_field_params, input_duration),
    FeedbackType.EARLY: FeedbackField(x, dx, feedback_field_params, input_duration),
    FeedbackType.LATE:  FeedbackField(x, dx, feedback_field_params, input_duration),
    FeedbackType.SWAP:  FeedbackField(x, dx, feedback_field_params, input_duration),
}


# ====================================
# -------- Experiment params ---------
# ====================================

t_feedback_lim = 30
t_feedback = np.arange(0, t_feedback_lim + dt, dt)
trigger_step = 100  # ~t=5, ~20% into feedback window

# Driver feedback examples:
# "Skip gym on Mondays"
# human_feedback = (FeedbackType.SKIP, "gym")

# "I arrive at work earlier now"
# human_feedback = (FeedbackType.EARLY, "work")

# "I go to gym later on Mondays"
# human_feedback = (FeedbackType.LATE, "gym")

# "Always predict home as final destination"
# human_feedback = (FeedbackType.LOCK, "home")

# "I go to gym before work now, not after"
human_feedback = (FeedbackType.SWAP, "work", "gym")


h_dmem_mask = np.ones(len(x))  # mask for locked destinations


# ====================================
# -------- Main experiment loop ------
# ====================================

for iteration in range(2):
    print(f"\n{'='*60}")
    print(f"Week {iteration + 4} (Iteration {iteration + 1})")
    print(f"{'='*60}")

    # ── reset fields ──────────────────────────────────────────────────
    u_dest, u_wm, h_u_dest = reset_execution_fields(u_dest_memory, h_d_initial, x)
    h_u_wm = h_0_wm * np.ones(len(x))
    reset_feedback_fields(feedback_fields)

    u_dest_history = []
    u_wm_history = []
    active_ff_history = []   # history of the active feedback field only

    # ── Loop 1: Execution ─────────────────────────────────────────────
    print("\nPhase 1: Executing Monday routine (prediction)...")

    for i in range(len(t)):
        conv_dest = compute_convolution(u_dest, theta_dest, w_hat_dest)
        conv_wm = compute_convolution(u_wm, theta_wm, w_hat_wm)
        f_dest = np.heaviside(u_dest - theta_dest, 1)
        f_wm = np.heaviside(u_wm - theta_wm, 1)

        h_u_dest += dt / tau_h_dest

        u_dest += dt * (-u_dest + conv_dest + input_destination_onset + h_u_dest - 6.0 * f_wm * conv_wm)
        u_wm += (dt / 1.25) * (-u_wm + conv_wm + 8 * (f_dest * u_dest) + h_u_wm)

        u_dest_history.append([u_dest[idx] for idx in destination_indices])
        u_wm_history.append([u_wm[idx] for idx in destination_indices])

    # ── Loop 2: Feedback window ───────────────────────────────────────
    print("\nPhase 2: Driver provides feedback...")

    feedback_type = human_feedback[0]
    destination_name = human_feedback[1]
    destination_center = resolve_feedback(destination_name, destination_names, destination_positions)
    ff = feedback_fields[feedback_type]

    # resolve second target for SWAP
    target_name = None
    target_center = None
    if feedback_type == FeedbackType.SWAP:
        target_name = human_feedback[2]
        target_center = resolve_feedback(target_name, destination_names, destination_positions)

    for i in range(len(t_feedback)):
        if i == trigger_step:
            print(f"  Driver says: '{feedback_type.value} {destination_name}'")
            ff.inject(center=destination_center, amplitude=3.0, width=5.0, current_step=i)
            if feedback_type == FeedbackType.SWAP:
                print(f"  (swap with: {target_name})")
                ff.inject_add(center=target_center, amplitude=3.0, width=5.0)

        # update all feedback fields
        for ftype, field in feedback_fields.items():
            if (field.params.transient and
                    field._inject_step is not None and
                    i >= field._inject_step + field.input_duration):
                field.clear_input()

            conv_ff = compute_convolution(field.u, field.params.theta, field.w_hat)
            field.u += (dt / field.params.tau) * (-field.u + conv_ff + field.h + field.s)

        active_ff_history.append(ff.u.copy())

    # ── Apply correction ──────────────────────────────────────────────
    print("\nPhase 3: Applying correction to destination memory...")
    u_dest_memory_before = u_dest_memory.copy()

    if feedback_type == FeedbackType.SKIP:
        u_dest_memory = apply_skip_from_field(u_dest_memory, feedback_fields[FeedbackType.SKIP])
        print(f"  → {destination_name} removed from routine")
    elif feedback_type == FeedbackType.LOCK:
        h_dmem_mask = apply_lock_from_field(h_dmem_mask, feedback_fields[FeedbackType.LOCK])
        print(f"  → {destination_name} locked (protected from future changes)")
    elif feedback_type == FeedbackType.EARLY:
        u_dest_memory = apply_early_from_field(u_dest_memory, feedback_fields[FeedbackType.EARLY], theta_dest)
        print(f"  → {destination_name} peak weakened (will be recalled earlier)")
    elif feedback_type == FeedbackType.LATE:
        u_dest_memory = apply_late_from_field(u_dest_memory, feedback_fields[FeedbackType.LATE], theta_dest)
        print(f"  → {destination_name} peak strengthened (will be recalled later)")
    elif feedback_type == FeedbackType.SWAP:
        u_dest_memory = apply_swap_from_field(u_dest_memory, feedback_fields[FeedbackType.SWAP])
        print(f"  → {destination_name} ↔ {target_name} order reversed")

    # ── Save ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(results_dir / f"u_dest_memory_week{iteration+4}_{timestamp}.npy", u_dest_memory)
    np.save(results_dir / f"h_dmem_mask_week{iteration+4}_{timestamp}.npy", h_dmem_mask)

    # Also save updated metadata
    corrected_metadata = metadata.copy()
    corrected_metadata['correction_applied'] = {
        'type': feedback_type.value,
        'destination': destination_name,
        'week': iteration + 4
    }
    np.save(results_dir / f"metadata_week{iteration+4}_{timestamp}.npy", corrected_metadata)

    # ── Plot ──────────────────────────────────────────────────────────
    active_ff_history = np.array(active_ff_history)
    target_idx = np.argmin(np.abs(x - destination_center))

    title_str = f"{feedback_type.value.upper()}: {destination_name}"
    if feedback_type == FeedbackType.SWAP:
        title_str += f" ↔ {target_name}"

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Week {iteration + 4} — Driver A Monday — {title_str}", fontsize=13)

    # Panel 1: Memory before/after
    for bucket_idx, color, name in zip(destination_buckets, destination_colors, destination_names):
        axs[0].plot(x[bucket_idx], u_dest_memory_before[bucket_idx],
                    '--', linewidth=2, color=color, alpha=0.6)
        axs[0].plot(x[bucket_idx], u_dest_memory[bucket_idx],
                    '-', linewidth=2.5, color=color, label=name)
    
    # Mark destinations
    for dest_pos, dest_name, color in zip(destination_positions, destination_names, destination_colors):
        axs[0].axvline(dest_pos, color=color, linestyle=':', alpha=0.3)
    
    axs[0].set_title("Destination Memory: before (--) vs after (-)")
    axs[0].set_xlabel("Destination space (x)")
    axs[0].set_ylabel("Memory activation")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, alpha=0.3)

    # Panel 2: Feedback field at target location over time
    axs[1].plot(t_feedback[:len(active_ff_history)], active_ff_history[:, target_idx], 
                label=f"{feedback_type.value} field", linewidth=2, color='purple')
    axs[1].axhline(feedback_field_params.theta, color='k', linestyle='--', 
                   linewidth=1, label='threshold', alpha=0.5)
    axs[1].axvline(trigger_step * dt, color='r', linestyle='--', 
                   linewidth=1, label='feedback given', alpha=0.7)
    axs[1].set_title(f"{feedback_type.value.upper()} field at {destination_name}")
    axs[1].set_xlabel("Feedback window time")
    axs[1].set_ylabel("Field activation")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Panel 3: Feedback field spatial profile at end
    axs[2].plot(x, ff.u, label="Field activation", linewidth=2, color='purple')
    axs[2].plot(x, ff.output(), linestyle='--', linewidth=2, 
                label="Thresholded output", color='orange')
    axs[2].axhline(feedback_field_params.theta, color='k', linestyle='--', 
                   linewidth=1, label='threshold', alpha=0.5)
    axs[2].axvline(destination_center, color='gray', linestyle=':', 
                   linewidth=2, label=destination_name)
    
    if feedback_type == FeedbackType.SWAP:
        axs[2].axvline(target_center, color='orange', linestyle=':', 
                       linewidth=2, label=target_name)
    
    # Mark all destinations
    for dest_pos in destination_positions:
        if dest_pos != destination_center and (feedback_type != FeedbackType.SWAP or dest_pos != target_center):
            axs[2].axvline(dest_pos, color='lightgray', linestyle=':', alpha=0.3)
    
    axs[2].set_title(f"{feedback_type.value.upper()} field spatial profile")
    axs[2].set_xlabel("Destination space (x)")
    axs[2].set_ylabel("Activation")
    axs[2].legend(fontsize=9)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / f"correction_week{iteration+4}_{timestamp}.png", dpi=150)
    plt.show()

    # ── Analysis ──────────────────────────────────────────────────────
    print("\nMemory peak analysis:")
    for i, (dest_name, dest_pos) in enumerate(zip(destination_names, destination_positions)):
        idx = destination_indices[i]
        amp_before = u_dest_memory_before[idx]
        amp_after = u_dest_memory[idx]
        change = amp_after - amp_before
        
        print(f"  {dest_name:8s}: {amp_before:.3f} → {amp_after:.3f} "
              f"(change: {change:+.3f})")

print("\n" + "="*60)
print("All correction iterations complete!")
print("="*60)