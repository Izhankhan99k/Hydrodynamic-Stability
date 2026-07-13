import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import os
import multiprocessing as mp
import time

# Verify: Force single-threaded matrix operations for safe multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

logging.getLogger('dedalus').setLevel(logging.WARNING)

def solve_evp_at_resolution(Pe, k_val, Ra, S, Lambda, d0, N):
    """
    Builds and solves the EVP for a specific resolution N.
    Returns the raw, scaled eigenvalues.
    """
    j = 1j
    Pr = 1.0        

    zcoord = d3.Coordinate('z')
    dist = d3.Distributor(zcoord, dtype=np.complex128)
    
    basis_l = d3.ChebyshevT(zcoord, size=N, bounds=(0, 1))
    basis_s = d3.ChebyshevT(zcoord, size=N, bounds=(1, 1 + d0))
    zl_grid = dist.local_grid(basis_l)

    U = dist.Field(name='U', bases=basis_l)
    U['g'] = 4 * zl_grid * (1 - zl_grid)
    
    Uyy = dist.Field(name='Uyy', bases=basis_l)
    Uyy['g'] = -2 * np.ones_like(zl_grid)

    w = dist.Field(name='w', bases=basis_l)
    wy = dist.Field(name='wy', bases=basis_l)
    wyy = dist.Field(name='wyy', bases=basis_l)
    wyyy = dist.Field(name='wyyy', bases=basis_l)
    theta_l = dist.Field(name='theta_l', bases=basis_l)
    theta_ly = dist.Field(name='theta_ly', bases=basis_l)
    theta_s = dist.Field(name='theta_s', bases=basis_s)
    theta_sy = dist.Field(name='theta_sy', bases=basis_s)
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

    dz = lambda A: d3.Differentiate(A, zcoord)
    dy = lambda A: 0.5 * dz(A)
    
    lift_l = lambda A: d3.Lift(A, basis_l.derivative_basis(1), -1)
    lift_s = lambda A: d3.Lift(A, basis_s.derivative_basis(1), -1)

    variables = [w, wy, wyy, wyyy, theta_l, theta_ly, theta_s, theta_sy, h,
                 tau_w1, tau_w2, tau_w3, tau_w4, tau_tl1, tau_tl2, tau_ts1, tau_ts2]
                 
    problem = d3.EVP(variables, eigenvalue=sigma, namespace=locals())

    problem.add_equation("dy(w) - wy + lift_l(tau_w1) = 0")
    problem.add_equation("dy(wy) - wyy + lift_l(tau_w2) = 0")
    problem.add_equation("dy(wyy) - wyyy + lift_l(tau_w3) = 0")
    problem.add_equation(
        "sigma*(wyy - k_val**2 * w) "
        "- Pr*(dy(wyyy) - 2*k_val**2 * wyy + k_val**4 * w) "
        "+ Pe*(j*k_val*U*(wyy - k_val**2 * w) - j*k_val*Uyy*w) "
        "+ (k_val**2 * Ra * Pr / Pe) * theta_l + lift_l(tau_w4) = 0"
    )
    
    problem.add_equation("dy(theta_l) - theta_ly + lift_l(tau_tl1) = 0")
    problem.add_equation(
        "sigma*theta_l - (dy(theta_ly) - k_val**2 * theta_l) "
        "+ Pe*(j*k_val*U*theta_l - w) + lift_l(tau_tl2) = 0"
    )
    
    problem.add_equation("dy(theta_s) - theta_sy + lift_s(tau_ts1) = 0")
    problem.add_equation("sigma*theta_s - (dy(theta_sy) - k_val**2 * theta_s) + lift_s(tau_ts2) = 0")
    
    problem.add_equation("sigma*h - (1 / (Lambda * S)) * (theta_sy(z=1) - theta_ly(z=1)) = 0")

    top_z = 1 + d0
    problem.add_equation("w(z=0) = 0")
    problem.add_equation("wy(z=0) = 0")
    problem.add_equation("theta_l(z=0) = 0")
    problem.add_equation(f"theta_s(z={top_z}) = 0")
    problem.add_equation("w(z=1) = 0")
    problem.add_equation("wy(z=1) + 2*j*k_val*h = 0")
    problem.add_equation("theta_l(z=1) - h = 0")
    problem.add_equation("theta_s(z=1) - h = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])

    evals = solver.eigenvalues
    finite_evals = evals[np.isfinite(evals)]
    
    # Return scaled eigenvalues
    return finite_evals / Pe

def get_coupled_max_growth(Pe, k_val, Ra, S, Lambda, d0, N1=64, N2=80):
    """
    Solves the EVP at two different resolutions and filters out spurious numerical
    eigenvalues by measuring drift. Returns the max real physical growth rate.
    """
    # 1. Solve at base resolution
    evals_N1 = solve_evp_at_resolution(Pe, k_val, Ra, S, Lambda, d0, N=N1)
    
    # 2. Solve at higher resolution
    evals_N2 = solve_evp_at_resolution(Pe, k_val, Ra, S, Lambda, d0, N=N2)
    
    # 3. Two-grid drift filter
    drift_tolerance = 1e-5
    physical_growth_rates = []
    
    for val1 in evals_N1:
        # Find the shortest distance to any eigenvalue in the N2 solution
        drift = np.min(np.abs(evals_N2 - val1))
        
        # If the eigenvalue didn't move much, it's a physical mode
        if drift < drift_tolerance:
            physical_growth_rates.append(val1.real)
            
    # 4. Return the most unstable (or least stable) physical growth rate
    if len(physical_growth_rates) > 0:
        return np.max(physical_growth_rates)
    else:
        # Fallback if no physical modes are found (should not happen in normal bounds)
        return -10.0


def compute_growth_grid(Pe_array, k_array, S, Ra=0.0, Lambda=1.0, d0=1.0, N1=64, N2=80):
    num_Pe = len(Pe_array)
    num_k = len(k_array)
    Pe_grid, k_grid = np.meshgrid(Pe_array, k_array)
    growth_grid = np.zeros_like(Pe_grid)
    
    total = num_Pe * num_k
    count = 0
    for i in range(num_k):
        for j in range(num_Pe):
            growth_grid[i, j] = get_coupled_max_growth(
                Pe=Pe_grid[i, j], k_val=k_grid[i, j],
                Ra=Ra, S=S, Lambda=Lambda, d0=d0, N1=N1, N2=N2
            )
            count += 1
            if count % 25 == 0:
                print(f"  S={S}: {count}/{total} solves", flush=True)
    return Pe_grid, k_grid, growth_grid


def compute_for_S(args):
    S, Pe_array, k_array, Ra, Lambda, d0, N1, N2 = args
    pid = os.getpid()
    print(f"[PID {pid}] Starting S={S}", flush=True)
    t0 = time.time()
    
    Pe_grid, k_grid, growth_grid = compute_growth_grid(
        Pe_array, k_array, S,
        Ra=Ra, Lambda=Lambda, d0=d0, N1=N1, N2=N2
    )
    
    elapsed = time.time() - t0
    print(f"[PID {pid}] Finished S={S} in {elapsed:.1f} s", flush=True)
    
    filename = f"stability_data_S_{S}.npz"
    np.savez_compressed(filename, Pe_grid=Pe_grid, k_grid=k_grid, growth_grid=growth_grid)
    return S, Pe_grid, k_grid, growth_grid


if __name__ == "__main__":
    USER_Ra = 0.0
    USER_Lambda = 1.0
    USER_d0 = 1.0
    
    # ─── Resolution Settings ───
    # N1 and N2 must be different to check for convergence.
    # Higher N = more accuracy but significantly slower compute times.
    Res_1 = 64
    Res_2 = 80 
    
    S_values = [0.01, 0.1, 1, 10]
    num_Pe = 10
    num_k = 10
    
    Pe_array = np.linspace(2000, 15000, num_Pe)
    k_array = np.linspace(0.5, 1.8, num_k)
    
    args_list = [(S, Pe_array, k_array, USER_Ra, USER_Lambda, USER_d0, Res_1, Res_2) for S in S_values]
    
    n_procs = min(len(S_values), mp.cpu_count())
    print(f"Using {n_procs} processes for parallel solving.", flush=True)
    
    with mp.Pool(processes=n_procs) as pool:
        results = []
        for result in pool.imap_unordered(compute_for_S, args_list):
            results.append(result)
            print(f"Received result for S={result[0]}", flush=True)
            
    results.sort(key=lambda x: x[0])
    
    print("Consolidating all results into a master file...", flush=True)
    master_dict = {}
    for S, Pe_grid, k_grid, growth_grid in results:
        master_dict[f'Pe_grid_S_{S}'] = Pe_grid
        master_dict[f'k_grid_S_{S}'] = k_grid
        master_dict[f'growth_grid_S_{S}'] = growth_grid
        
    np.savez_compressed('master_neutral_stability_data.npz', **master_dict)
    print("Master data saved successfully as 'master_neutral_stability_data.npz'", flush=True)
    
    plt.figure(figsize=(11, 8))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(S_values)))
    
    legend_handles = []
    
    for idx, (S, Pe_grid, k_grid, growth_grid) in enumerate(results):
        plt.contour(Pe_grid, k_grid, growth_grid, levels=[0.0],
                    colors=[colors[idx]], linewidths=2.5)
        
        proxy_line, = plt.plot([], [], color=colors[idx], linewidth=2.5, label=f'S = {S}')
        legend_handles.append(proxy_line)

    plt.title(f'Neutral stability curves (Two-Grid Filtered)\nRa={USER_Ra}, Pr=1.0, Λ={USER_Lambda}, d0={USER_d0}',
              fontsize=15, pad=15)
    plt.xlabel('Peclet Number ($Pe$) [Literature Scale]', fontsize=13)
    plt.ylabel('Wavenumber ($k$) [Literature Scale]', fontsize=13)
    plt.xlim(Pe_array[0], Pe_array[-1])
    plt.ylim(k_array[0], k_array[-1])
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.legend(handles=legend_handles, loc='best', fontsize=11)
    
    plt.text(Pe_array[-1]*0.8, np.mean(k_array)*1.2, "UNSTABLE", 
             color='darkred', fontsize=14, fontweight='bold', ha='center')
    plt.text(Pe_array[0]*1.2, np.mean(k_array)*1.2, "STABLE", 
             color='steelblue', fontsize=14, fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.show()