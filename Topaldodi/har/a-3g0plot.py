import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors
# --- 1. LOAD BACKGROUND MAP DATA ---
# This loads the grid you already calculated
data = np.load("a-3g0.npz")
k_grid = data['k_grid']
Re_grid = data['Re_grid']
max_ci_grid = data['max_ci_grid']

a = data['a']
G = data['G']
Bo = data['Bo']



# --- 3. GENERATE THE VISUALIZATION ---
fig, ax = plt.subplots(figsize=(7, 5.5))

# Mask stable configurations to leave them completely white
masked_ci = np.ma.masked_where(max_ci_grid <= 0, max_ci_grid)

c_min = 1e-4
c_max = np.nanmax(masked_ci)

# 2. Force Matplotlib to create exactly 200 slices logarithmically spaced
smooth_log_levels = np.logspace(np.log10(c_min), np.log10(c_max), 100)

# 3. Plot with the explicit levels array
contour_fill = ax.contourf(k_grid, Re_grid, masked_ci, 
                           levels=smooth_log_levels, 
                           cmap='YlGnBu_r', 
                           norm=colors.LogNorm(vmin=c_min, vmax=c_max), 
                           alpha=0.9)

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

# --- 6. CUSTOM LEGEND ---
# Combine the contour patch and the scatter points into one clean legend
this_work_patch = mpatches.Patch(color="#C04343", label="Unstable Region")
# We grab the scatter plot handle automatically by relying on the label we assigned earlier
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, this_work_patch) 

ax.legend(handles=handles, loc='lower left', fontsize=11, frameon=True, edgecolor='gray')

plt.tight_layout()
plt.savefig('a-1g0.png', dpi=300)
plt.show()