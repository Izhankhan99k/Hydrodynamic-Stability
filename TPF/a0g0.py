import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata  # For smoothing the existing grid

# --- 1. LOAD BACKGROUND MAP DATA ---
data = np.load("a3g0.npz")
k_grid = data['k_grid']
Re_grid = data['Re_grid']
max_ci_grid = data['max_ci_grid']

a = data['a']
G = data['G']
# Bo = data['Bo'] # Unused in title, can leave commented if not needed

# --- 2. SMOOTHING BLOCK (Fixes the uneven/stair-case look) ---
# --- 2. SMOOTHING BLOCK (Fixes the uneven/stair-case look) ---
log_k_coarse = np.log10(k_grid.flatten())
log_Re_coarse = np.log10(Re_grid.flatten())
ci_coarse = max_ci_grid.flatten()

log_k_fine_1d = np.linspace(-2, 2, 300) 
log_Re_fine_1d = np.linspace(-1, 5, 300) 
log_k_fine_mesh, log_Re_fine_mesh = np.meshgrid(log_k_fine_1d, log_Re_fine_1d)

# CHANGED: Switch 'cubic' to 'linear' to stop the boundary from bleeding/oscillating
smoothed_ci = griddata(
    (log_k_coarse, log_Re_coarse), ci_coarse, 
    (log_k_fine_mesh, log_Re_fine_mesh), method='linear'
)

# Convert back to physical dimensions for plotting
k_grid_smooth = 10**log_k_fine_mesh
Re_grid_smooth = 10**log_Re_fine_mesh

# --- 3. GENERATE THE VISUALIZATION ---
fig, ax = plt.subplots(figsize=(7.5, 5.5)) # Slightly widened to accommodate colorbar cleanly

# Mask stable configurations to leave them completely white
masked_ci = np.ma.masked_where(smoothed_ci <= 0, smoothed_ci)

c_min = 1e-4
c_max = np.nanmax(masked_ci)

# Force Matplotlib to create exactly 100 slices logarithmically spaced
smooth_log_levels = np.logspace(np.log10(c_min), np.log10(c_max), 100)

# Plot with the smooth high-res data
contour_fill = ax.contourf(k_grid_smooth, Re_grid_smooth, masked_ci, 
                           levels=smooth_log_levels, 
                           cmap='YlGnBu_r', 
                           norm=colors.LogNorm(vmin=c_min, vmax=c_max), 
                           alpha=0.9)

# --- 4. COLORBAR (Replaces the misleading red legend patch) ---
cbar = fig.colorbar(contour_fill, ax=ax, pad=0.03, aspect=25)
cbar.set_label(r'Growth Rate ($max(c_i)$)', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)

# --- 5. AXIS SCALING AND FORMATTING ---
ax.set_xscale('log')
ax.set_yscale('log')

# Explicitly set axis limits
ax.set_xlim(1e-2, 1e2)
ax.set_ylim(1e-1, 1e5)

# Minimalistic professional tick spacing adjustments
ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
ax.set_yticks([1e-1, 1e1, 1e3, 1e5])

# Axis labels
ax.set_xlabel(r'$k$', fontsize=15, style='italic', labelpad=5)
ax.set_ylabel(r'$Re$', fontsize=15, style='italic', rotation=0, labelpad=15)
ax.set_title(rf'$a = {a}, G = {G}$', fontsize=14, pad=10)

# Tick adjustments
ax.tick_params(axis='both', which='major', labelsize=12, direction='in', 
                length=6, top=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', 
                length=3, top=True, right=True)

ax.patch.set_facecolor('white')
plt.grid(False)

plt.tight_layout()
plt.savefig('a-1g0_smooth.png', dpi=300)
