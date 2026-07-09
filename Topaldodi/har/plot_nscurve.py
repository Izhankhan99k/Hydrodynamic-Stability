import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- LOAD DATA ---
data = np.load("stability_results.npz")
k_grid = data['k_grid']
Re_grid = data['Re_grid']
max_ci_grid = data['max_ci_grid']

a = data['a']
G = data['G']
Bo = data['Bo']

# --- GENERATE THE EXACT PAPER VISUALIZATION STYLE ---
fig, ax = plt.subplots(figsize=(7, 5.5))

# Mask stable configurations to leave them completely white, filling only the unstable region (ci > 0)
masked_ci = np.ma.masked_where(max_ci_grid <= 0, max_ci_grid)

# Using 'YlGn' (Yellow to Green transition) to perfectly match the color tone of the paper snippet
contour_fill = ax.contourf(k_grid, Re_grid, masked_ci, levels=15, cmap=plt.cm.YlGn, alpha=0.9)

# Draw the bold black/blue bounding Neutral Stability border line exactly at ci = 0
ax.contour(k_grid, Re_grid, max_ci_grid, levels=[0.0], colors=['#1E3A8A'], linewidths=[2.5], zorder=4)

# Set logarithmic scale for axes to match target dimensions
ax.set_xscale('log')
ax.set_yscale('log')

# Explicitly set axis limits to match your paper clipping window
ax.set_xlim(1e-2, 1e2)
ax.set_ylim(1e-1, 1e5)

# Minimalistic professional tick spacing adjustments
ax.set_xticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
ax.set_yticks([1e-1, 1e1, 1e3, 1e5])

# Axis labels matching the font format
ax.set_xlabel(r'$k$', fontsize=15, style='italic', labelpad=5)
ax.set_ylabel(r'$Re$', fontsize=15, style='italic', rotation=0, labelpad=15)
ax.set_title(rf'$a = {a}, G = {G}$', fontsize=14, pad=10)

# Minimalistic professional tick spacing adjustments (Updated for all 4 sides)
ax.tick_params(axis='both', which='major', labelsize=12, direction='in', 
               length=6, top=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', 
               length=3, top=True, right=True)

# Clean, scientific layout box styling
ax.patch.set_facecolor('white')
plt.grid(False) # Turn off grid lines to keep the clean look of the paper

# Custom Paper Legend Emulation matching the new palette
this_work_patch = mpatches.Patch(color="#C04343", label="Unstable Region")
ax.legend(handles=[this_work_patch], loc='lower left', fontsize=11, frameon=True, edgecolor='gray')

plt.tight_layout()
plt.savefig('nscurve.png', dpi=300)
plt.show()