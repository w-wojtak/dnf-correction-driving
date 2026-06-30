# #!/usr/bin/env python3
# import numpy as np
# import matplotlib.pyplot as plt
# import sys

# trial = int(sys.argv[1]) if len(sys.argv) > 1 else 1
# x_lim, dx = 80, 0.05
# x = np.arange(-x_lim, x_lim + dx, dx)

# filepath = f'C:/Users/weronika.wojtak/VSCodeProject/dnf-correction-driving/results_driving_correction/h_dmem_mask_cmd1_20260629_171750.npy'
# h_u_amem = np.load(filepath)

# plt.figure(figsize=(5, 4))
# plt.plot(x, h_u_amem, linewidth=2)
# plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
# plt.xlabel('Position (x)')
# plt.ylabel('Activation')
# plt.title(r'$m_{LOCK}$')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# print(f"Trial {trial} - Max: {np.max(h_u_amem):.4f}, Min: {np.min(h_u_amem):.4f}")
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

trial = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# -----------------------------------
# Spatial axis
# -----------------------------------
x_lim, dx = 80, 0.05
x = np.arange(-x_lim, x_lim + dx, dx)

# -----------------------------------
# Paths
# -----------------------------------
project_root = Path(r"C:/Users/weronika.wojtak/VSCodeProject/dnf-correction-driving")

mask_path = project_root / "results_driving_correction" / "h_dmem_mask_cmd1_20260629_171750.npy"

# Load the same metadata source used in sequence_tuning.py
metadata_files = sorted((project_root / "results_driving_learning").glob("metadata_driverA_monday_*.npy"))
if not metadata_files:
    raise FileNotFoundError("No metadata_driverA_monday_*.npy file found.")

metadata_path = metadata_files[-1]   # latest one
metadata = np.load(metadata_path, allow_pickle=True).item()

# -----------------------------------
# Load data
# -----------------------------------
h_dmem_mask = np.load(mask_path)

destination_positions = metadata["destination_positions"]
destination_names = metadata["destination_names"]

# Same colors idea as in your file
base_colors = ['tab:brown', 'tab:blue', 'tab:green', 'tab:red']
destination_colors = base_colors[:len(destination_names)]

# -----------------------------------
# Same bucket logic as sequence_tuning.py
# -----------------------------------
def compute_destination_bounds(x, positions):
    positions = np.array(positions)
    midpoints = (positions[:-1] + positions[1:]) / 2
    boundaries = np.concatenate(([x[0]], midpoints, [x[-1] + 1e-9]))
    return (
        [np.where((x >= boundaries[i]) & (x < boundaries[i + 1]))[0]
         for i in range(len(positions))],
        boundaries
    )

destination_buckets, boundaries = compute_destination_bounds(x, destination_positions)

# -----------------------------------
# Plot
# -----------------------------------
fig, ax = plt.subplots(figsize=(5, 4))

# # Optional: lightly shade each interval
# for i, (color, name) in enumerate(zip(destination_colors, destination_names)):
#     left = boundaries[i]
#     right = boundaries[i + 1]
#     ax.axvspan(left, right, color=color, alpha=0.08)

# Plot each interval in its own color, like panel 1
for bucket_idx, color, name in zip(destination_buckets, destination_colors, destination_names):
    ax.plot(
        x[bucket_idx],
        h_dmem_mask[bucket_idx],
        linewidth=2.5,
        color=color,
        label=name
    )

# Mark destination centers
for dest_pos, color in zip(destination_positions, destination_colors):
    ax.axvline(dest_pos, color=color, linestyle=':', alpha=0.35)

# Put labels above intervals
for i, (name, color) in enumerate(zip(destination_names, destination_colors)):
    left = boundaries[i]
    right = boundaries[i + 1]
    center = 0.5 * (left + right)
    ax.text(
        center, 1, name,
        transform=ax.get_xaxis_transform(),   # x in data coords, y in axes coords
        ha='center', va='top',
        color=color, fontsize=9, fontweight='bold'
    )

ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Position (x)')
ax.set_ylabel('Activation')
ax.set_title(r'$m_{LOCK}$')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Trial {trial} - Max: {np.max(h_dmem_mask):.4f}, Min: {np.min(h_dmem_mask):.4f}")