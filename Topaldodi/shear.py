import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
CW = MPI.COMM_WORLD
rank = CW.rank
size = CW.size
def solve_evp_rescaled(Ra, Pe, Pr, S, Lam, k, N=64):
    """
    Rescaled velocity W = Pe * w to remove the 1/Pe singularity at Pe=0.
    """
    k2 = k**2
    ik = 1j * k
    Lam2 = Lam**2
    Lam2_k2 = Lam2 * k2
    coeff_s = 1.0 / (Lam2 * S)
    coeff_l = 1.0 / (Lam * S)
    buoy_rescaled = k2 * Ra * Pr  # Replaced k2*Ra*Pr/Pe with k2*Ra*Pr
    zcoord = d3.Coordinate('z')
    dist = d3.Distributor(zcoord, dtype=np.complex128)
    zbasis = d3.Chebyshev(zcoord, size=N, bounds=(0, 1))
    z = dist.local_grid(zbasis)
    U = dist.Field(name='U', bases=zbasis)
    U['g'] = 1.0 - z
    W        = dist.Field(name='W',        bases=zbasis)
    Wz       = dist.Field(name='Wz',       bases=zbasis)
    eta      = dist.Field(name='eta',      bases=zbasis)
    etaz     = dist.Field(name='etaz',     bases=zbasis)
    theta_l  = dist.Field(name='theta_l',  bases=zbasis)
    theta_lz = dist.Field(name='theta_lz', bases=zbasis)
    theta_s  = dist.Field(name='theta_s',  bases=zbasis)
    theta_sz = dist.Field(name='theta_sz', bases=zbasis)
    h_hat    = dist.Field(name='h_hat')
    sigma    = dist.Field(name='sigma')
    tau_W1   = dist.Field(name='tau_W1')
    tau_W2   = dist.Field(name='tau_W2')
    tau_eta1 = dist.Field(name='tau_eta1')
    tau_eta2 = dist.Field(name='tau_eta2')
    tau_tl1  = dist.Field(name='tau_tl1')
    tau_tl2  = dist.Field(name='tau_tl2')
    tau_ts1  = dist.Field(name='tau_ts1')
    tau_ts2  = dist.Field(name='tau_ts2')
    dz = lambda A: d3.Differentiate(A, zcoord)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    problem = d3.EVP(
        [W, Wz, eta, etaz, theta_l, theta_lz, theta_s, theta_sz, h_hat,
         tau_W1, tau_W2, tau_eta1, tau_eta2, tau_tl1, tau_tl2, tau_ts1, tau_ts2],
        eigenvalue=sigma, namespace=locals()
    )
    problem.add_equation("dz(W) - Wz + lift(tau_W1, -1) = 0")
    problem.add_equation("dz(eta) - etaz + lift(tau_eta1, -1) = 0")
    problem.add_equation("dz(theta_l) - theta_lz + lift(tau_tl1, -1) = 0")
    problem.add_equation("dz(theta_s) - theta_sz + lift(tau_ts1, -1) = 0")
    problem.add_equation("dz(Wz) - k2*W - eta + lift(tau_W2, -1) = 0")
    problem.add_equation("sigma*eta + Pr*(dz(etaz) - k2*eta) - ik*Pe*U*eta - buoy_rescaled*theta_l + lift(tau_eta2, -1) = 0")
    problem.add_equation("sigma*theta_l + dz(theta_lz) - k2*theta_l - ik*Pe*U*theta_l + W + lift(tau_tl2, -1) = 0")
    problem.add_equation("Lam2*sigma*theta_s + dz(theta_sz) - Lam2_k2*theta_s + lift(tau_ts2, -1) = 0")
    problem.add_equation("-sigma*h_hat - coeff_s*theta_sz(z=0) + coeff_l*theta_lz(z=1) = 0")
    problem.add_equation("W(z=0) = 0")
    problem.add_equation("Wz(z=0) = 0")
    problem.add_equation("theta_l(z=0) = 0")
    problem.add_equation("W(z=1) = 0")
    problem.add_equation("Wz(z=1) + ik*Pe*h_hat = 0")
    problem.add_equation("theta_l(z=1) - h_hat = 0")
    problem.add_equation("theta_s(z=0) - h_hat = 0")
    problem.add_equation("theta_s(z=1) = 0")
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return solver.eigenvalues
def filter_eigenvalues(evals, cutoff=1e6):
    good = np.isfinite(evals) & (np.abs(evals) < cutoff)
    return evals[good]
def filter_by_resolution_test(evals1, evals2, tolerance=1e-4):
    resolved = []
    for e1 in evals1:
        dists = np.abs(evals2 - e1)
        if len(dists) > 0 and np.min(dists) < tolerance:
            resolved.append(e1)
    return np.array(resolved)
if __name__ == "__main__":
    Pr = 7.0    # Typical for water
    S = 0.01    # Typical Stefan number in these problems
    Lam = 0.1   # d0 = 0.1
    Ra = 1700.0
    N = 64
    tol = 1e-4
    k_vals = np.linspace(0.1, 7.0, 40)
    Pe_list = [0.0, 0.15, 0.50, 2.00, 5.00]
    
    local_k_vals = k_vals[rank::size]
    local_growth_data = {}
    for Pe in Pe_list:
        local_rates = []
        for kv in local_k_vals:
            ev_N_raw = solve_evp_rescaled(Ra, Pe, Pr, S, Lam, kv, N)
            ev_N = filter_eigenvalues(ev_N_raw)
            ev_N16_raw = solve_evp_rescaled(Ra, Pe, Pr, S, Lam, kv, N+16)
            ev_N16 = filter_eigenvalues(ev_N16_raw)
            ev_res = filter_by_resolution_test(ev_N, ev_N16, tolerance=tol)
            
            if len(ev_res) > 0:
                sig = ev_res[np.argmin(ev_res.real)]
                local_rates.append(-sig.real)
            else:
                local_rates.append(np.nan)
        local_growth_data[Pe] = np.array(local_rates)
        
        all_rates = CW.gather(local_growth_data[Pe], root=0)
        
        if rank == 0:
            full_rates = np.zeros(len(k_vals))
            for r in range(size):
                full_rates[r::size] = all_rates[r]
            local_growth_data[Pe] = full_rates
            print(f"Pe = {Pe:.2f} done. Max growth rate: {np.nanmax(full_rates):.4f}")
    if rank == 0:
        plt.figure(figsize=(9, 6))
        for Pe in Pe_list:
            plt.plot(k_vals, local_growth_data[Pe], '-o', markersize=4, lw=2, label=f'Pe = {Pe}')
        plt.axhline(0, color='grey', ls='--', lw=1)
        plt.xlabel('Wavenumber $k$', fontsize=14)
        plt.ylabel(r'Real growth rate $\sigma_r$', fontsize=14)
        plt.title(f'Shear Stabilization of Phase Boundary (Ra={Ra}, $d_0$={Lam})', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('shear_stabilization.png', dpi=200)
        print("Saved shear_stabilization.png")