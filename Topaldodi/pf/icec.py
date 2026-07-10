import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import time

# Suppress logging to keep the terminal progress tracker clean
logging.getLogger('dedalus').setLevel(logging.WARNING)

def get_eigenvalues(Pe, k_val, Ra, S, Lambda, d0, N):
    """
    Solves the fully coupled multiphysics EVP and returns all scaled eigenvalues 
    for a given resolution N.
    """
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
    U['g'] = 1 - zl_grid**2
    Uzz = dist.Field(name='Uzz', bases=basis_l)
    Uzz['g'] = -2 * np.ones_like(zl_grid)

    # Fields
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

    # Tau polynomials
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
    problem.add_equation("w(z=-1) = 0")
    problem.add_equation("wz(z=-1) = 0")
    problem.add_equation("theta_l(z=-1) = 0")
    problem.add_equation(f"theta_s(z={top_z}) = 0")
    problem.add_equation("w(z=1) = 0")
    problem.add_equation("wz(z=1) + 4*j*k_val*h = 0")
    problem.add_equation("theta_l(z=1) - h = 0")
    problem.add_equation("theta_s(z=1) - h = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])

    # Extract, filter infinites/NaNs, and scale back to advective timescale
    evals = solver.eigenvalues
    finite_evals = evals[np.isfinite(evals)]
    return finite_evals / Pe


def get_clean_eigenspectrum(Pe, k_val, Ra, S, Lambda, d0, N1=150, N2=160, tol=1e-5):
    """
    Solves the EVP at two different resolutions (N1, N2).
    Cross-references the complex eigenvalues. If an eigenvalue drifts by more 
    than 'tol' between N1 and N2, it is flagged as spurious and removed.
    """
    print(f"Solving EVP at N={N1}...")
    evals_N1 = get_eigenvalues(Pe, k_val, Ra, S, Lambda, d0, N1)
    
    print(f"Solving EVP at N={N2}...")
    evals_N2 = get_eigenvalues(Pe, k_val, Ra, S, Lambda, d0, N2)

    # Cross-reference: Calculate the distance matrix between all N1 and N2 eigenvalues
    # NumPy broadcasting creates an array of shape (len(evals_N1), len(evals_N2))
    diffs = np.abs(evals_N1[:, np.newaxis] - evals_N2[np.newaxis, :])
    
    # Find the minimum distance to any N2 eigenvalue for each N1 eigenvalue
    min_diffs = np.min(diffs, axis=1)

    # Keep only the N1 eigenvalues that have a stationary match in N2
    clean_mask = min_diffs < tol
    clean_evals = evals_N1[clean_mask]

    print(f"Filtering complete: Kept {len(clean_evals)} physical modes out of {len(evals_N1)} total modes.")

    # Convert to physical quantities
    raw_gr = evals_N1.real
    raw_ps = -evals_N1.imag / k_val
    
    clean_gr = clean_evals.real
    clean_ps = -clean_evals.imag / k_val

    return raw_gr, raw_ps, clean_gr, clean_ps


# =========================================================================
# Execution: Eigenspectrum Plotting
# =========================================================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # USER CONFIGURATION
    # ---------------------------------------------------------
    TARGET_Pe = 15000       
    TARGET_k = 2.0          
    USER_Ra = 0.0           
    USER_S = 1e6            
    USER_Lambda = 1.0       
    USER_d0 = 1.0           
    
    RES_1 = 140
    RES_2 = 150
    TOLERANCE = 1e-4
    # ---------------------------------------------------------
    
    print(f"--- Generating Cleaned Eigenspectrum ---")
    start_time = time.time()
    
    raw_gr, raw_ps, clean_gr, clean_ps = get_clean_eigenspectrum(
        Pe=TARGET_Pe, 
        k_val=TARGET_k, 
        Ra=USER_Ra, 
        S=USER_S, 
        Lambda=USER_Lambda, 
        d0=USER_d0, 
        N1=RES_1, 
        N2=RES_2,
        tol=TOLERANCE
    )
    
    elapsed = time.time() - start_time
    print(f"Total solve time: {elapsed:.2f} seconds.")

    # --- Visualization ---
    plt.figure(figsize=(9, 7))
    
    # 1. Plot the discarded (spurious) modes as faded points in the background
    plt.scatter(raw_gr, raw_ps, color='lightgray', alpha=0.5, marker='x', 
                label='Spurious (Discarded) Modes')
    
    # 2. Plot the converged (physical) modes clearly
    plt.scatter(clean_gr, clean_ps, color='crimson', edgecolors='k', s=60, zorder=5, 
                label='Physical (Converged) Modes')
    
    # Reference lines
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5, label='Neutral Stability ($\sigma_r = 0$)')
    plt.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    
    # Restrict axes to view the actual physics, otherwise the plot gets skewed by 
    # highly negative spurious values outside our viewing window.
    plt.xlim(-100.0, 100.0) 
    plt.ylim(-5.0, 5.0)
    
    # Formatting
    plt.title(f'Converged Eigenspectrum ($Pe={TARGET_Pe}$, $k={TARGET_k}$)\nFiltered via $N_1={RES_1}$, $N_2={RES_2}$ drift check', 
              fontsize=14, pad=15)
    plt.xlabel('Growth Rate $\sigma_r$', fontsize=13)
    plt.ylabel('Phase Speed ($- \sigma_i / k$)', fontsize=13)
    
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.show()