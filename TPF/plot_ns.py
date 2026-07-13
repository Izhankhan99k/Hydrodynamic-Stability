import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = np.load('master_neutral_stability_data.npz')
s_values = [key.replace('Pe_grid_S_', '') for key in data.files if key.startswith('Pe_grid_S_')]
s_values.sort(key=float)

plt.figure(figsize=(10, 7))
colors = cm.jet(np.linspace(0, 1, len(s_values)))

for s, color in zip(s_values, colors):
    Pe = data[f'Pe_grid_S_{s}']/2
    k = data[f'k_grid_S_{s}']/2
    growth = data[f'growth_grid_S_{s}']
    
    cs = plt.contour(Pe, k, growth, levels=[0], colors=[color], linewidths=2)
    plt.plot([], [], color=color, label=f'S = {s}', linewidth=2)

 # often Pe is log-scaled
plt.xlabel('Peclet Number ($Pe$)', fontsize=14)
plt.ylabel('Wavenumber ($k$)', fontsize=14)
plt.title('Neutral Stability Curves for Different Stefan Numbers', fontsize=16)
plt.legend(title='Stefan Number ($S$)', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.grid(True, which="both", ls=":", alpha=0.7)
plt.tight_layout()
plt.savefig('neutral_stability_plot.png', dpi=300)
plt.show()
print("Final plot saved.")