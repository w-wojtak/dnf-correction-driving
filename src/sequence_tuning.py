# sequence_tuning.py
"""
DNF-based sequence correction with natural language interface.
Complete standalone implementation.
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from dataclasses import dataclass
from feedback_types import FeedbackType


# ====================================
# -------- Feedback Field Classes ----
# ====================================

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
results_dir = PROJECT_ROOT / "results_driving_correction"
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


early_late_coupling = 0.5

def apply_early_from_field(u_dest_memory, early_field, theta_dest):
    """Weaken memory peak where early field is active → recalls earlier."""
    u_new = u_dest_memory.copy()
    u_new += early_late_coupling * early_field.output() * np.heaviside(u_dest_memory - theta_dest, 1.0)
    return u_new


def apply_late_from_field(u_dest_memory, late_field, theta_dest):
    """Strengthen memory peak where late field is active → recalls later."""
    u_new = u_dest_memory.copy()
    u_new -= early_late_coupling * late_field.output() * np.heaviside(u_dest_memory - theta_dest, 1.0)
    return u_new


def apply_swap_from_field(u_dest_memory, swap_field):
    """Swap memory values between two active regions of swap field."""
    u_new = u_dest_memory.copy()
    active = swap_field.output() > 0

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

input_duration = 50
t_feedback_lim = 30
t_feedback = np.arange(0, t_feedback_lim + dt, dt)
trigger_step = 100

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

h_dmem_mask = np.ones(len(x))



# ====================================
# -------- Visualization -------------
# ====================================

def plot_correction(u_before, u_after, ff, feedback_type, destination_name, 
                   target_name=None, save_path=None):
    """
    Visualize the correction process.
    
    Args:
        u_before: Memory field before correction
        u_after: Memory field after correction
        ff: FeedbackField that was used
        feedback_type: FeedbackType enum
        destination_name: Primary target destination
        target_name: Secondary target (for SWAP)
        save_path: Optional path to save figure
    """
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    title_str = f"{feedback_type.value.upper()}: {destination_name}"
    if target_name:
        title_str += f" ↔ {target_name}"
    fig.suptitle(title_str, fontsize=14, fontweight='bold')
    
    # Panel 1: Memory before/after
    for bucket_idx, color, name in zip(destination_buckets, destination_colors, destination_names):
        axs[0].plot(x[bucket_idx], u_before[bucket_idx],
                    '--', linewidth=2, color=color, alpha=0.6, label=f"{name} (before)")
        axs[0].plot(x[bucket_idx], u_after[bucket_idx],
                    '-', linewidth=2.5, color=color, label=f"{name} (after)")
    
    # Mark destinations
    for dest_pos, dest_name, color in zip(destination_positions, destination_names, destination_colors):
        axs[0].axvline(dest_pos, color=color, linestyle=':', alpha=0.3)
    
    axs[0].axhline(theta_dest, color='k', linestyle='--', alpha=0.3, label='threshold')
    axs[0].set_title("Memory Field: Before vs After")
    axs[0].set_xlabel("Spatial position (x)")
    axs[0].set_ylabel("Activation")
    axs[0].legend(loc='upper right', fontsize=8)
    axs[0].grid(True, alpha=0.3)
    
    # Panel 2: Feedback field spatial profile
    axs[1].plot(x, ff.u, linewidth=2, color='purple', label='Field activation')
    axs[1].plot(x, ff.output(), linestyle='--', linewidth=2, 
                color='orange', label='Thresholded output')
    axs[1].axhline(ff.params.theta, color='k', linestyle='--', 
                   linewidth=1, alpha=0.5, label='threshold')
    
    # Mark target destination(s)
    target_center = resolve_feedback(destination_name, destination_names, destination_positions)
    axs[1].axvline(target_center, color='red', linestyle=':', 
                   linewidth=2, label=destination_name)
    
    if target_name:
        target2_center = resolve_feedback(target_name, destination_names, destination_positions)
        axs[1].axvline(target2_center, color='orange', linestyle=':', 
                       linewidth=2, label=target_name)
    
    # Mark all destinations faintly
    for dest_pos in destination_positions:
        if dest_pos != target_center and (not target_name or dest_pos != target2_center):
            axs[1].axvline(dest_pos, color='lightgray', linestyle=':', alpha=0.3)
    
    axs[1].set_title(f"{feedback_type.value.upper()} Feedback Field")
    axs[1].set_xlabel("Spatial position (x)")
    axs[1].set_ylabel("Activation")
    axs[1].legend(fontsize=9)
    axs[1].grid(True, alpha=0.3)
    
    # Panel 3: Memory amplitude comparison (bar chart)
    dest_names_short = destination_names
    before_amps = [u_before[idx] for idx in destination_indices]
    after_amps = [u_after[idx] for idx in destination_indices]
    
    x_pos = np.arange(len(dest_names_short))
    width = 0.35
    
    bars1 = axs[2].bar(x_pos - width/2, before_amps, width, 
                       label='Before', color='gray', alpha=0.6)
    bars2 = axs[2].bar(x_pos + width/2, after_amps, width, 
                       label='After', color='blue', alpha=0.8)
    
    axs[2].axhline(theta_dest, color='k', linestyle='--', alpha=0.3, label='threshold')
    axs[2].set_xlabel('Destination')
    axs[2].set_ylabel('Peak Amplitude')
    axs[2].set_title('Memory Peak Comparison')
    axs[2].set_xticks(x_pos)
    axs[2].set_xticklabels(dest_names_short)
    axs[2].legend()
    axs[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axs[2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved plot to {save_path}")
    
    plt.show()


# ====================================
# -------- Natural Language Experiments
# ====================================

if __name__ == "__main__":
    from llm_parser import LLMParser
    from dotenv import load_dotenv
    import os
    
    # Load API key from .env
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    
    print("\n" + "="*70)
    print("NATURAL LANGUAGE DNF CORRECTION")
    print("="*70)
    
    if GROQ_API_KEY:
        print("✅ Using Groq API (Llama 3.1)")
    else:
        print("ℹ️  Using rule-based parser")
    
    # ========================================================================
    # EDIT THESE COMMANDS TO TEST DIFFERENT CORRECTIONS
    # ========================================================================
    
    user_commands = [
        # "Skip gym on Mondays",
        "I won't go to the gym anymore",
        # "I go to gym before work now",
        # "I arrive at work earlier now",
        # "I arrive at work later now",
        # "I arrive at the gym later now",
        # "Swap order of work and gym",
    ]
    
    # ========================================================================
    # Process each command
    # ========================================================================
    
    for cmd_idx, user_command in enumerate(user_commands):
        print(f"\n{'='*70}")
        print(f"COMMAND {cmd_idx + 1}/{len(user_commands)}")
        print(f"{'='*70}")
        print(f"💬 User: '{user_command}'")
        
        # Initialize parser
        parser = LLMParser(api_key=GROQ_API_KEY, use_api=(GROQ_API_KEY is not None))
        
        # Parse natural language
        parsed = parser.parse_command(user_command, destination_names)
        
        if parsed['feedback_type'] is None:
            print(f"❌ Could not parse command, skipping")
            continue
        
        print(f"🤖 Parsed: {parsed['feedback_type'].value.upper()}")
        print(f"   Target: {parsed['target']}")
        if parsed.get('target2'):
            print(f"   Target2: {parsed['target2']}")
        
        # Convert to tuple format
        if parsed['feedback_type'] == FeedbackType.SWAP:
            human_feedback = (parsed['feedback_type'], parsed['target'], parsed['target2'])
        else:
            human_feedback = (parsed['feedback_type'], parsed['target'])
        
        # Extract feedback info
        feedback_type = human_feedback[0]
        destination_name = human_feedback[1]
        destination_center = resolve_feedback(destination_name, destination_names, destination_positions)
        ff = feedback_fields[feedback_type]
        
        # Handle SWAP second target
        target_name = None
        target_center = None
        if feedback_type == FeedbackType.SWAP:
            target_name = human_feedback[2]
            target_center = resolve_feedback(target_name, destination_names, destination_positions)
        
        # Run feedback field dynamics
        print(f"\n⚙️  Running feedback field dynamics...")
        for i in range(len(t_feedback)):
            if i == trigger_step:
                ff.inject(center=destination_center, amplitude=3.0, width=5.0, current_step=i)
                if feedback_type == FeedbackType.SWAP:
                    ff.inject_add(center=target_center, amplitude=3.0, width=5.0)
            
            # Update all feedback fields
            for ftype, field in feedback_fields.items():
                if (field.params.transient and
                        field._inject_step is not None and
                        i >= field._inject_step + field.input_duration):
                    field.clear_input()
                
                conv_ff = compute_convolution(field.u, field.params.theta, field.w_hat)
                field.u += (dt / field.params.tau) * (-field.u + conv_ff + field.h + field.s)
        
        # Apply correction to memory
        print(f"✏️  Applying correction to memory...")
        u_dest_memory_before = u_dest_memory.copy()
        
        if feedback_type == FeedbackType.SKIP:
            u_dest_memory = apply_skip_from_field(u_dest_memory, feedback_fields[FeedbackType.SKIP])
            print(f"   → Removed {destination_name} from routine")
            
        elif feedback_type == FeedbackType.LOCK:
            h_dmem_mask = apply_lock_from_field(h_dmem_mask, feedback_fields[FeedbackType.LOCK])
            print(f"   → Locked {destination_name} (protected from changes)")
            
        elif feedback_type == FeedbackType.EARLY:
            u_dest_memory = apply_early_from_field(u_dest_memory, feedback_fields[FeedbackType.EARLY], theta_dest)
            print(f"   → {destination_name} peak strengthened (will recall earlier)")
            
        elif feedback_type == FeedbackType.LATE:
            u_dest_memory = apply_late_from_field(u_dest_memory, feedback_fields[FeedbackType.LATE], theta_dest)
            print(f"   → {destination_name} peak weakened (will recall later)")
            
        elif feedback_type == FeedbackType.SWAP:
            u_dest_memory = apply_swap_from_field(u_dest_memory, feedback_fields[FeedbackType.SWAP])
            print(f"   → Swapped {destination_name} ↔ {target_name}")
        
        # Reset feedback field
        ff.clear_input()
        ff.u = ff.params.h_0 * np.ones(len(x))


        
        # Show memory changes
        print(f"\n📊 Memory changes:")
        for i, (dest_name, dest_pos) in enumerate(zip(destination_names, destination_positions)):
            idx = destination_indices[i]
            amp_before = u_dest_memory_before[idx]
            amp_after = u_dest_memory[idx]
            change = amp_after - amp_before
            
            if abs(change) > 1e-6:
                print(f"   {dest_name:8s}: {amp_before:.3f} → {amp_after:.3f} (Δ {change:+.3f}) ⚠️")
            else:
                print(f"   {dest_name:8s}: {amp_before:.3f} → {amp_after:.3f} (Δ {change:+.3f})")
        
        # ============================================================
        # ADD THIS: Plot the correction
        # ============================================================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n📈 Generating visualization...")
        save_path = results_dir / f"correction_cmd{cmd_idx+1}_{feedback_type.value}_{timestamp}.png"
        plot_correction(
            u_before=u_dest_memory_before,
            u_after=u_dest_memory,
            ff=ff,
            feedback_type=feedback_type,
            destination_name=destination_name,
            target_name=target_name,
            save_path=save_path
        )
        # ============================================================
        
        # Save state

        np.save(results_dir / f"u_dest_memory_cmd{cmd_idx+1}_{timestamp}.npy", u_dest_memory)
        np.save(results_dir / f"h_dmem_mask_cmd{cmd_idx+1}_{timestamp}.npy", h_dmem_mask)
        
        # # Show memory changes
        # print(f"\n📊 Memory changes:")
        # for i, (dest_name, dest_pos) in enumerate(zip(destination_names, destination_positions)):
        #     idx = destination_indices[i]
        #     amp_before = u_dest_memory_before[idx]
        #     amp_after = u_dest_memory[idx]
        #     change = amp_after - amp_before
            
        #     if abs(change) > 1e-6:
        #         print(f"   {dest_name:8s}: {amp_before:.3f} → {amp_after:.3f} (Δ {change:+.3f}) ⚠️")
        #     else:
        #         print(f"   {dest_name:8s}: {amp_before:.3f} → {amp_after:.3f} (Δ {change:+.3f})")



        
        
        # # Save state
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # np.save(results_dir / f"u_dest_memory_cmd{cmd_idx+1}_{timestamp}.npy", u_dest_memory)
        # np.save(results_dir / f"h_dmem_mask_cmd{cmd_idx+1}_{timestamp}.npy", h_dmem_mask)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ All {len(user_commands)} corrections complete!")
    print(f"{'='*70}")
    print(f"\nFinal routine state:")
    for i, (dest_name, dest_pos) in enumerate(zip(destination_names, destination_positions)):
        idx = destination_indices[i]
        amp = u_dest_memory[idx]
        if amp > theta_dest:
            print(f"  ✓ {dest_name:8s}: amplitude {amp:.3f}")
        else:
            print(f"  ✗ {dest_name:8s}: amplitude {amp:.3f} (removed)")