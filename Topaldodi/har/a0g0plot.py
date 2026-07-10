import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 1. LOAD BACKGROUND MAP DATA ---
# This loads the grid you already calculated
data = np.load("neutral_stability_map.npz")
k_grid = data['k_grid']
Re_grid = data['Re_grid']
max_ci_grid = data['max_ci_grid']

a = data['a']
G = data['G']
Bo = data['Bo']

# --- 2. LOAD YOUR CSV DATA ---
# np.loadtxt is perfect here since your CSV doesn't have text headers
true_data = np.loadtxt("true_data.csv", delimiter=",")
k_true = true_data[:, 0]
Re_true = true_data[:, 1]

# --- 3. GENERATE THE VISUALIZATION ---
fig, ax = plt.subplots(figsize=(7, 5.5))

# Mask stable configurations to leave them completely white
masked_ci = np.ma.masked_where(max_ci_grid <= 0, max_ci_grid)

# Plot the background contour (Yellow to Green transition)
contour_fill = ax.contourf(k_grid, Re_grid, masked_ci, levels=15, cmap=plt.cm.YlOrBr, alpha=0.9)



# --- 4. PLOT YOUR TRUE CSV DATA ON TOP ---
# Using solid black dots so they stand out clearly against the green/yellow background
ax.scatter(k_true, Re_true, color='black', marker='o', s=25, zorder=5, label='True Data (CSV)')


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
plt.savefig('nscurve_with_data.png', dpi=300)
plt.show()