import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import time

# Suppress logging to keep the terminal progress tracker clean
logging.getLogger('dedalus').setLevel(logging.WARNING)

def get_coupled_max_growth(Pe, k_val, Ra, S, Lambda, d0, N1=50, N2=64, tol=1e-3):
    """
    Applies the two-grid resolution method. Solves the EVP at N1 and N2,
    filters out spurious modes based on a convergence tolerance, 
    and returns the absolute maximum growth rate without ad hoc physical bounds.
    """
    # 1. Solve at both resolutions
    ev1 = solve_evp_raw(Pe, k_val, Ra, S, Lambda, d0, N1)
    ev2 = solve_evp_raw(Pe, k_val, Ra, S, Lambda, d0, N2)

    # 2. Convergence Filter (Cross-matching)
    physical_evals = []
    for e1 in ev1:
        # Calculate the distance from this eigenvalue to ALL high-res eigenvalues
        drift = np.min(np.abs(ev2 - e1))
        
        # Relative tolerance check (avoids rejecting large valid eigenvalues)
        denom = max(abs(e1), 1.0)
        if (drift / denom) < tol:
            physical_evals.append(e1)

    physical_evals = np.array(physical_evals)

    # If all modes were filtered out (the spectrum is completely unstable/spurious)
    if len(physical_evals) == 0:
        return -1.0  # Safe floor value for plotting the stable region

    # 3. Extract the absolute maximum growth rate directly
    growth_rates = physical_evals.real
    
    return np.max(growth_rates)
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # USER CONFIGURATION
    # ---------------------------------------------------------
    USER_Ra = 0.0       # Rayleigh number
    USER_S = 1e6        # Stefan number (Frozen limit)
    USER_Lambda = 1.0   
    USER_d0 = 1.0       
    # ---------------------------------------------------------
    
    print(f"--- Generating Coupled Neutral Stability Curve ---")
    
    # Base Resolution Grid (Adjusted to 40x40 for testing speed)
    # Total solves = num_Pe * num_k * 2
    num_Pe = 40
    num_k = 40
    
    # Define the viewport for the sweep
    Pe_array = np.linspace(2000, 25000, num_Pe)
    k_array = np.linspace(0.5, 3.5, num_k)
    
    Pe_grid, k_grid = np.meshgrid(Pe_array, k_array)
    growth_grid = np.zeros_like(Pe_grid)
    
    total_points = num_Pe * num_k
    start_time = time.time()
    
    # Sweep the Parameter Space
    point_count = 0
    for i in range(num_k):
        for j in range(num_Pe):
            
            # Call the new wrapper that handles N1, N2, and filtering automatically
            growth_grid[i, j] = get_coupled_max_growth(
                Pe=Pe_grid[i, j], 
                k_val=k_grid[i, j], 
                Ra=USER_Ra, 
                S=USER_S, 
                Lambda=USER_Lambda, 
                d0=USER_d0, 
                N1=50,       # Base resolution
                N2=64,       # High resolution
                tol=1e-3     # Filter tolerance
            )
            
            point_count += 1
            if point_count % 25 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {point_count}/{total_points} grid points completed ({elapsed:.1f} sec)")

    print("Sweep complete. Generating high-resolution plot...")

    # 3. Visualization
    plt.figure(figsize=(11, 8))
    
    # Fill the unstable regime
    plt.contourf(Pe_grid, k_grid, growth_grid, levels=[0.0, 100], 
                 colors=['crimson'], alpha=0.15)
    
    # Draw the Neutral Curve (High resolution will make this perfectly smooth)
    cs = plt.contour(Pe_grid, k_grid, growth_grid, levels=[0.0], 
                     colors='crimson', linewidths=2.5)
    
    # Add text labels
    plt.text(np.max(Pe_array)*0.8, np.mean(k_array), "UNSTABLE\n($\sigma_r > 0$)", 
             color='darkred', fontsize=14, fontweight='bold', ha='center', va='center')
    plt.text(np.min(Pe_array)*1.2, np.mean(k_array), "STABLE\n($\sigma_r < 0$)", 
             color='steelblue', fontsize=14, fontweight='bold', ha='center', va='center')

    # Formatting
    plt.title(f'Cleaned Coupled Marginal Stability Curve ({num_Pe}x{num_k})\nRa={USER_Ra}, S={USER_S}, Pr=1.0', 
              fontsize=15, pad=15)
    plt.xlabel('Peclet Number ($Pe$)', fontsize=13)
    plt.ylabel('Wavenumber ($k$)', fontsize=13)
    
    plt.xlim(np.min(Pe_array), np.max(Pe_array))
    plt.ylim(np.min(k_array), np.max(k_array))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.show()