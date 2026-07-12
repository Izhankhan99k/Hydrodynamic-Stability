import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import griddata

# --- 1. LOAD BACKGROUND MAP DATA ---
data = np.load("a3g0.npz")
k_grid = data['k_grid']
Re_grid = data['Re_grid']
max_ci_grid = data['max_ci_grid']  # Raw data containing negative/stable values

a = data['a']
G = data['G']

# --- 2. HIGH-RES SMOOTHING BLOCK (Interpolate first, then mask) ---
# Flatten the raw coordinates and data points
log_k_coarse = np.log10(k_grid.flatten())
log_Re_coarse = np.log10(Re_grid.flatten())
ci_coarse = max_ci_grid.flatten()  # DO NOT mask yet!

# Create a highly dense grid in log-space (300x300 points)
log_k_fine_1d = np.linspace(-2, 2, 300)   # spans 1e-2 to 1e2
log_Re_fine_1d = np.linspace(-1, 5, 300)  # spans 1e-1 to 1e5
log_k_fine_mesh, log_Re_fine_mesh = np.meshgrid(log_k_fine_1d, log_Re_fine_1d)

# Interpolate the continuous raw values (linear prevents bleeding/floating islands)
smoothed_ci_raw = griddata(
    (log_k_coarse, log_Re_coarse), ci_coarse, 
    (log_k_fine_mesh, log_Re_fine_mesh), method='linear'
)

# Convert coordinates back to physical space for plotting
k_grid_smooth = 10**log_k_fine_mesh
Re_grid_smooth = 10**log_Re_fine_mesh

# NOW mask the high-resolution data to leave stable regions white
# (Optional: change 0 to 1e-5 if your data contains microscopic solver noise)
masked_ci_smooth = np.ma.masked_where(smoothed_ci_raw <= 0, smoothed_ci_raw)

# --- 3. GENERATE THE VISUALIZATION ---
fig, ax = plt.subplots(figsize=(7.5, 5.5))

c_min = 1e-4
c_max = np.nanmax(masked_ci_smooth)

# Create 100 log-spaced contour levels based on the smooth data limits
smooth_log_levels = np.logspace(np.log10(c_min), np.log10(c_max), 100)

# Plot using the high-resolution grid and high-resolution mask
contour_fill = ax.contourf(k_grid_smooth, Re_grid_smooth, masked_ci_smooth, 
                           levels=smooth_log_levels, 
                           cmap='YlGnBu_r', 
                           norm=colors.LogNorm(vmin=c_min, vmax=c_max), 
                           alpha=0.9)

# --- 4. COLORBAR ---
cbar = fig.colorbar(contour_fill, ax=ax, pad=0.03, aspect=25)
cbar.set_label(r'Growth Rate ($\max(c_i)$)', fontsize=12, labelpad=10)
cbar.ax.tick_params(labelsize=10)

# --- 5. AXIS SCALING AND FORMATTING ---
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(1e-2, 1e2)
ax.set_ylim(1e-1, 1e5)

ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
ax.set_yticks([1e-1, 1e1, 1e3, 1e5])

ax.set_xlabel(r'$k$', fontsize=15, style='italic', labelpad=5)
ax.set_ylabel(r'$Re$', fontsize=15, style='italic', rotation=0, labelpad=15)
ax.set_title(rf'$a = {a}, G = {G}$', fontsize=14, pad=10)

ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6, top=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', length=3, top=True, right=True)

ax.patch.set_facecolor('white')
plt.grid(False)

plt.tight_layout()
plt.savefig('a-1g0_perfectly_smooth.png', dpi=300)
plt.show()