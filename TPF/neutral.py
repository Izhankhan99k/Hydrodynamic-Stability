import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import os
import multiprocessing as mp
import time

logging.getLogger('dedalus').setLevel(logging.WARNING)

def get_coupled_max_growth(Pe, k_val, Ra, S, Lambda, d0, N=64):
    """
    Solves the fully coupled multiphysics EVP for a specific Pe and k.
    Returns the maximum physical growth rate scaled to the advective timescale.
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
    Uzz = dist.Field(name='Uzz', bases=basis_l)
    Uzz['g'] = -8 * np.ones_like(zl_grid)

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

    dz = lambda A: d3.Differentiate(A, zcoord)
    lift_l = lambda A: d3.Lift(A, basis_l.derivative_basis(1), -1)
    lift_s = lambda A: d3.Lift(A, basis_s.derivative_basis(1), -1)

    variables = [w, wz, wzz, wzzz, theta_l, theta_lz, theta_s, theta_sz, h,
                 tau_w1, tau_w2, tau_w3, tau_w4, tau_tl1, tau_tl2, tau_ts1, tau_ts2]
                 
    problem = d3.EVP(variables, eigenvalue=sigma, namespace=locals())

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
    finite_evals = evals[np.isfinite(evals)]
    scaled_evals = finite_evals / Pe
    
    growth_rates = scaled_evals.real
    phase_speeds = -scaled_evals.imag / k_val

    mask = (growth_rates > -1.5) & (phase_speeds > -0.5) & (phase_speeds < 1.5)
    clean_growth = growth_rates[mask]
    
    if len(clean_growth) > 0:
        return np.max(clean_growth)
    else:
        return -1.0


def compute_growth_grid(Pe_array, k_array, S, Ra=0.0, Lambda=1.0, d0=1.0, N=64):
    """Sweep over (Pe, k) for a given S and return the growth matrix."""
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
                Ra=Ra, S=S, Lambda=Lambda, d0=d0, N=N
            )
            count += 1
            if count % 50 == 0:
                print(f"  S={S}: {count}/{total} solves", flush=True)
    return Pe_grid, k_grid, growth_grid

def compute_for_S(args):
    """Wrapper that unpacks arguments, calls grid compute, and saves backup."""
    S, Pe_array, k_array, Ra, Lambda, d0, N = args
    pid = os.getpid()
    print(f"[PID {pid}] Starting S={S}", flush=True)
    t0 = time.time()
    Pe_grid, k_grid, growth_grid = compute_growth_grid(
        Pe_array, k_array, S,
        Ra=Ra, Lambda=Lambda, d0=d0, N=N
    )
    elapsed = time.time() - t0
    print(f"[PID {pid}] Finished S={S} in {elapsed:.1f} s", flush=True)
    
    # Backup checkpoint save per process
    filename = f"stability_data_S_{S}.npz"
    np.savez_compressed(filename, Pe_grid=Pe_grid, k_grid=k_grid, growth_grid=growth_grid)
    return S, Pe_grid, k_grid, growth_grid


if __name__ == "__main__":
    # Parameters
    USER_Ra = 0.0
    USER_Lambda = 1.0
    USER_d0 = 1.0
    N_res = 64
    S_values = [0.00001,0.0001,0.001,0.01,0.1 ,1,10]
    num_Pe = 80
    num_k = 80
    Pe_array = np.linspace(2000, 25000, num_Pe)
    k_array = np.linspace(0.5, 3.5, num_k)
    
    args_list = [(S, Pe_array, k_array, USER_Ra, USER_Lambda, USER_d0, N_res) for S in S_values]
    
    n_procs = min(len(S_values), mp.cpu_count())
    print(f"Using {n_procs} processes.", flush=True)
    
    with mp.Pool(processes=n_procs) as pool:
        results = []
        for result in pool.imap_unordered(compute_for_S, args_list):
            results.append(result)
            print(f"Received result for S={result[0]}", flush=True)
            
    # ═══ FIX 1: Sort results by S to guarantee consistent color mapping ═══
    results.sort(key=lambda x: x[0])
    
    print("Consolidating all results into a master file...", flush=True)
    master_dict = {}
    for S, Pe_grid, k_grid, growth_grid in results:
        master_dict[f'Pe_grid_S_{S}'] = Pe_grid
        master_dict[f'k_grid_S_{S}'] = k_grid
        master_dict[f'growth_grid_S_{S}'] = growth_grid
        
    np.savez_compressed('master_neutral_stability_data.npz', **master_dict)
    print("Master data saved successfully as 'master_neutral_stability_data.npz'", flush=True)
    
    # Plotting Setup
    plt.figure(figsize=(11, 8))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(S_values)))
    
    legend_handles = []
    
    for idx, (S, Pe_grid, k_grid, growth_grid) in enumerate(results):
        # Generate the contour line
        plt.contour(Pe_grid, k_grid, growth_grid, levels=[0.0],
                    colors=[colors[idx]], linewidths=2.5)
        
        # ═══ FIX 2: Use independent line proxies for a bulletproof legend ═══
        proxy_line, = plt.plot([], [], color=colors[idx], linewidth=2.5, label=f'S = {S}')
        legend_handles.append(proxy_line)

    # Formatting
    plt.title(f'Neutral stability curves for different Stefan numbers\nRa={USER_Ra}, Pr=1.0, Λ={USER_Lambda}, d0={USER_d0}',
              fontsize=15, pad=15)
    plt.xlabel('Peclet Number ($Pe$)', fontsize=13)
    plt.ylabel('Wavenumber ($k$)', fontsize=13)
    plt.xlim(Pe_array[0], Pe_array[-1])
    plt.ylim(k_array[0], k_array[-1])
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Use the reliable proxy handle list explicitly
    plt.legend(handles=legend_handles, loc='best', fontsize=11)
    
    plt.text(Pe_array[-1]*0.8, np.mean(k_array)*1.2, "UNSTABLE", 
             color='darkred', fontsize=14, fontweight='bold', ha='center')
    plt.text(Pe_array[0]*1.2, np.mean(k_array)*1.2, "STABLE", 
             color='steelblue', fontsize=14, fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.show()