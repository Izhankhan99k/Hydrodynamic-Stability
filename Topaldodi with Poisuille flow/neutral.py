import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import dedalus.public as d3
import logging
import time
import os

# Suppress logging to keep the terminal progress tracker clean
logging.getLogger('dedalus').setLevel(logging.WARNING)

def solve_evp_raw(Pe, k_val, Ra, S, Lambda, d0, N):
    """Builds and solves the EVP for a given resolution N."""
    j = 1j
    Pr = 1.0        

    zcoord = d3.Coordinate('z')
    dist = d3.Distributor(zcoord, dtype=np.complex128)
    
    # Dual domains for Liquid and Solid
    basis_l = d3.ChebyshevT(zcoord, size=N, bounds=(0, 1))
    basis_s = d3.ChebyshevT(zcoord, size=N, bounds=(1, 1 + d0))
    zl_grid = dist.local_grid(basis_l)

    # Base State (Liquid)
    U = dist.Field(name='U', bases=basis_l)
    U['g'] = 4 * zl_grid * (1 - zl_grid)
    Uzz = dist.Field(name='Uzz', bases=basis_l)
    Uzz['g'] = -8 * np.ones_like(zl_grid)

    # Fields & Taus
    w = dist.Field(name='w', bases=basis_l)
    wz = dist.Field(name='wz', bases=basis_l)
    wzz = dist.Field(name='wzz', bases=basis_l)
    wzzz = dist.Field(name='wzzz', bases=basis_l)
    theta_l = dist.Field(name='theta_l', bases=basis_l)
    theta_lz = dist.Field(name='theta_lz', bases=basis_l)
    theta_s = dist.Field(name='theta_s', bases=basis_s)
    theta_sz = dist.Field(name='theta_sz', bases=basis_s)
    h = dist.Field(name='h')
    sigma = dist.Field(name='sigma')

    tau_w1 = dist.Field(name='tau_w1')
    tau_w2 = dist.Field(name='tau_w2')
    tau_w3 = dist.Field(name='tau_w3')
    tau_w4 = dist.Field(name='tau_w4')
    tau_tl1 = dist.Field(name='tau_tl1')
    tau_tl2 = dist.Field(name='tau_tl2')
    tau_ts1 = dist.Field(name='tau_ts1')
    tau_ts2 = dist.Field(name='tau_ts2')

    # Operators
    dz = lambda A: d3.Differentiate(A, zcoord)
    lift_l = lambda A: d3.Lift(A, basis_l.derivative_basis(1), -1)
    lift_s = lambda A: d3.Lift(A, basis_s.derivative_basis(1), -1)

    variables = [w, wz, wzz, wzzz, theta_l, theta_lz, theta_s, theta_sz, h,
                 tau_w1, tau_w2, tau_w3, tau_w4, tau_tl1, tau_tl2, tau_ts1, tau_ts2]
                 
    problem = d3.EVP(variables, eigenvalue=sigma, namespace=locals())

    # --- Equations ---
    problem.add_equation("dz(w) - wz + lift_l(tau_w1) = 0")
    problem.add_equation("dz(wz) - wzz + lift_l(tau_w2) = 0")
    problem.add_equation("dz(wzz) - wzzz + lift_l(tau_w3) = 0")
    problem.add_equation(
        "sigma*(wzz - k_val**2 * w) "
        "- Pr*(dz(wzzz) - 2*k_val**2 * wzz + k_val**4 * w) "
        "+ Pe*(j*k_val*U*(wzz - k_val**2 * w) - j*k_val*Uzz*w) "
        "+ (k_val**2 * Ra * Pr / Pe) * theta_l + lift_l(tau_w4) = 0"
    )
    problem.add_equation("dz(theta_l) - theta_lz + lift_l(tau_tl1) = 0")
    problem.add_equation(
        "sigma*theta_l - (dz(theta_lz) - k_val**2 * theta_l) "
        "+ Pe*(j*k_val*U*theta_l - w) + lift_l(tau_tl2) = 0"
    )
    problem.add_equation("dz(theta_s) - theta_sz + lift_s(tau_ts1) = 0")
    problem.add_equation("sigma*theta_s - (dz(theta_sz) - k_val**2 * theta_s) + lift_s(tau_ts2) = 0")
    problem.add_equation("sigma*h - (1 / (Lambda * S)) * (theta_sz(z=1) - theta_lz(z=1)) = 0")

    # --- Boundary Conditions ---
    top_z = 1 + d0
    problem.add_equation("w(z=0) = 0")
    problem.add_equation("wz(z=0) = 0")
    problem.add_equation("theta_l(z=0) = 0")
    problem.add_equation(f"theta_s(z={top_z}) = 0")
    problem.add_equation("w(z=1) = 0")
    problem.add_equation("wz(z=1) + 4*j*k_val*h = 0")
    problem.add_equation("theta_l(z=1) - h = 0")
    problem.add_equation("theta_s(z=1) - h = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])

    evals = solver.eigenvalues
    return evals[np.isfinite(evals)] / Pe


def get_coupled_max_growth(Pe, k_val, Ra, S, Lambda, d0, N1=40, N2=56, tol=1e-3):
    """Two-grid resolution filter to extract the absolute max physical growth rate."""
    ev1 = solve_evp_raw(Pe, k_val, Ra, S, Lambda, d0, N1)
    ev2 = solve_evp_raw(Pe, k_val, Ra, S, Lambda, d0, N2)

    physical_evals = []
    for e1 in ev1:
        drift = np.min(np.abs(ev2 - e1))
        denom = max(abs(e1), 1.0)
        if (drift / denom) < tol:
            physical_evals.append(e1)

    physical_evals = np.array(physical_evals)

    if len(physical_evals) == 0:
        return -1.0  # Safe floor value for stable region if no modes converge

    return np.max(physical_evals.real)


# =========================================================================
# Execution: Multi-Curve Grid Sweeping & Plotting
# =========================================================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # PARAMETER SETUP
    # ---------------------------------------------------------
    USER_Ra = 0.0       
    USER_Lambda = 1.0   
    USER_d0 = 1.0       
    
    # The array of Stefan numbers you requested
    S_values = [1, 5, 10, 100, 1000, 10000, 100000]
    
    # Grid Resolution (Keep moderate to avoid infinite compute times)
    num_Pe = 40
    num_k = 40
    
    Pe_array = np.linspace(2000, 25000, num_Pe)
    k_array = np.linspace(0.5, 3.5, num_k)
    Pe_grid, k_grid = np.meshgrid(Pe_array, k_array)
    
    total_points = num_Pe * num_k
    
    # ---------------------------------------------------------
    # GENERATE A COLOR GRADIENT FOR THE LINES
    # ---------------------------------------------------------
    # 'plasma' is excellent for sequential lines (goes from purple to yellow)
    color_map = plt.cm.plasma(np.linspace(0, 0.9, len(S_values)))
    
    # Dictionary to hold the data grids so we can plot them all at the end
    results_dict = {}

    print(f"--- Starting Multi-Curve Neutral Stability Sweep ---")
    print(f"Total iterations planned: {len(S_values)} curves x {total_points} grid points = {len(S_values)*total_points} total points.")
    
    global_start_time = time.time()
    
    # ---------------------------------------------------------
    # MASTER LOOP OVER STEFAN NUMBERS
    # ---------------------------------------------------------
    for idx, S in enumerate(S_values):
        print(f"\n[{idx+1}/{len(S_values)}] Sweeping grid for Stefan Number (S) = {S} ...")
        growth_grid = np.zeros_like(Pe_grid)
        
        point_count = 0
        loop_start = time.time()
        
        for i in range(num_k):
            for j in range(num_Pe):
                
                # Notice N1 and N2 are slightly lowered (40, 56) to drastically speed up execution 
                # while maintaining enough spectral density to filter spurious modes.
                growth_grid[i, j] = get_coupled_max_growth(
                    Pe=Pe_grid[i, j], 
                    k_val=k_grid[i, j], 
                    Ra=USER_Ra, 
                    S=S, 
                    Lambda=USER_Lambda, 
                    d0=USER_d0, 
                    N1=40, N2=56, tol=1e-3
                )
                
                point_count += 1
                if point_count % 100 == 0:
                    elapsed = time.time() - loop_start
                    print(f"  -> {point_count}/{total_points} points completed ({elapsed:.1f} sec)")

        # Store the completed grid in our dictionary
        results_dict[S] = growth_grid
        
        # Save a backup file just in case the script crashes during the next loop!
        np.savez(f'backup_S_{S}.npz', Pe_grid=Pe_grid, k_grid=k_grid, growth_grid=growth_grid)
        print(f"Finished S = {S}. Backup saved to 'backup_S_{S}.npz'.")

    total_time = time.time() - global_start_time
    print(f"\nAll sweeps complete in {total_time/60:.1f} minutes. Generating final plot...")

    # ---------------------------------------------------------
    # MASTER VISUALIZATION
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    
    legend_elements = []
    
    # Plot each neutral curve (0.0 contour level)
    for idx, S in enumerate(S_values):
        color = color_map[idx]
        grid_data = results_dict[S]
        
        # Draw the curve
        cs = ax.contour(Pe_grid, k_grid, grid_data, levels=[0.0], colors=[color], linewidths=2.0)
        
        # Create a custom legend proxy artist for this specific line
        legend_elements.append(Line2D([0], [0], color=color, lw=2.5, label=f'$S = {S}$'))

    # Formatting
    ax.set_title(f'Neutral Stability Curves for Varying Stefan Numbers\nRa={USER_Ra}, $\Lambda$={USER_Lambda}, $Pr=1.0$', 
              fontsize=14, pad=15)
    ax.set_xlabel('Peclet Number ($Pe$)', fontsize=13)
    ax.set_ylabel('Wavenumber ($k$)', fontsize=13)
    
    ax.set_xlim(np.min(Pe_array), np.max(Pe_array))
    ax.set_ylim(np.min(k_array), np.max(k_array))
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Attach the custom legend
    ax.legend(handles=legend_elements, loc='best', fontsize=11, frameon=True, edgecolor='gray', title="Stefan Number")
    
    plt.tight_layout()
    plt.savefig('multi_stefan_neutral_curves.png', dpi=300)
    plt.show()
    #.