import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the saved data
data = np.load("stability_results.npz")

cr_clean = data['cr_clean']
ci_clean = data['ci_clean']
unstable_mask = data['unstable_mask']
stable_mask = data['stable_mask']

# Load parameters for the title
Re = data['Re']
k = data['k']
a = data['a']
G = data['G']
Bo = data['Bo']

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot stable modes (original open blue circles styling)
ax.scatter(cr_clean[stable_mask], ci_clean[stable_mask], 
           marker='o', facecolors='none', edgecolors='blue', label='Stable Eigenmodes')

# Plot unstable modes as solid red dots (ci > 0)
if np.any(unstable_mask):
    ax.scatter(cr_clean[unstable_mask], ci_clean[unstable_mask], 
               color='red', marker='o', s=60, edgecolors='darkred', zorder=4, 
               label='Unstable Modes ($c_i > 0$)')

# Highlight the single most unstable mode with a gold star
if len(cr_clean) > 0:
    idx = np.argmax(ci_clean)
    ax.scatter(cr_clean[idx], ci_clean[idx], color='silver', marker='*', s=250, 
               edgecolors='k', zorder=5, label='Most Unstable Mode')
    
    ax.annotate(f"({cr_clean[idx]:.3f}, {ci_clean[idx]:.3f})", 
                 xy=(cr_clean[idx], ci_clean[idx]), xycoords='data',
                 xytext=(15, 10), textcoords='offset points',
                 fontsize=11, fontweight='bold', color='darkred',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='gray'))
    
    logger.info(f"Most unstable mode: c = {cr_clean[idx]:.4f} {ci_clean[idx]:+.4f}i")

# Plot styling adjustments
ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax.set_xlabel(r'Phase Speed ($c_r$)', fontsize=14)
ax.set_ylabel(r'Growth Rate ($c_i$)', fontsize=14)
ax.set_title(f'Couette-Poiseuille Free-Surface Spectrum\n$Re={Re}, k={k}, a={a}, G={G}, Bo={Bo}$', fontsize=14)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right')
plt.xlim(-1, 2.5)
plt.ylim(-2.5, 1)
plt.tight_layout()
plt.show()