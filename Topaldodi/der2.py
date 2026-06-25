"""
============================================================================
Eigenvalue Solver: Coupled Shear–Buoyancy–Phase-Change Stability
============================================================================
Dedalus v3  |  Toppaladoddi & Wettlaufer, JFM 2019
Solves the full coupled system:
  1. Orr-Sommerfeld equation      (liquid, 4th order → vorticity split)
  2. Energy equation              (liquid, 2nd order)
  3. Energy equation              (solid, 2nd order, mapped coordinate)
  4. Stefan condition             (scalar, interface dynamics)
Perturbation convention:  ~ exp(ikx − σt)
  Re(σ) > 0 → stable   (perturbation decays)
  Re(σ) < 0 → unstable (perturbation grows)
  Growth rate = −Re(σ)
Physical setup:
  Liquid: z  ∈ [0, 1],     Couette flow U(z) = 1−z, heated from below
  Solid:  ζ  ∈ [1, 1+Λ],   pure conduction, cooled from above
  Interface:  z = 1,       at the melting temperature (θ = 0)
Implementation:
  The solid is mapped to z_s ∈ [0, 1] via ζ = 1 + Λ·z_s.
  Both domains share the SAME Chebyshev basis on [0, 1].
  This keeps σ LINEAR everywhere → standard generalized EVP.
First-order reduction (Dedalus pattern):
  η ≡ (D²−k²)w    is the z-vorticity (NOT pressure)
  wz = Dw,  ηz = Dη,  θ_lz = Dθ_l,  θ_sz = D_{z_s} θ_s
Run:
  Single core:   python dedalus_evp.py
  MPI parallel:  mpiexec -n 4 python dedalus_evp.py
============================================================================
"""
import numpy as np
import dedalus.public as d3
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe for MPI)
import matplotlib.pyplot as plt
import logging
from mpi4py import MPI
logger = logging.getLogger(__name__)
# ── MPI setup ────────────────────────────────────────────────────────────
CW   = MPI.COMM_WORLD
rank = CW.rank
size = CW.size
# ======================================================================
# SOLVER
# ======================================================================
def solve_evp(Ra, Pe, Pr, S, Lam, k, N=64):
    """
    Solve the coupled liquid–solid eigenvalue problem.
    Parameters
    ----------
    Ra  : float – Rayleigh number
    Pe  : float – Peclet number (> 0)
    Pr  : float – Prandtl number
    S   : float – Stefan number
    Lam : float – Λ = (T_m − T_c) / (T_h − T_m), solid/liquid depth ratio
    k   : float – streamwise wavenumber (k > 0)
    N   : int   – Chebyshev resolution (same for both domains)
    Returns
    -------
    eigenvalues : 1-D complex array (σ values)
    solver      : Dedalus solver object (for eigenfunction extraction)
    """
    # -- derived constants ------------------------------------------------
    k2      = k ** 2
    ik      = 1j * k
    Lam2    = Lam ** 2
    buoy    = k2 * Ra * Pr / Pe          # buoyancy coupling coefficient
    Lam2_k2 = Lam2 * k2                  # Λ²k² in mapped solid equation
    coeff_s = 1.0 / (Lam2 * S)           # solid flux coeff in Stefan
    coeff_l = 1.0 / (Lam * S)            # liquid flux coeff in Stefan
    # -- coordinate and basis ---------------------------------------------
    #    Both the liquid (physical z ∈ [0,1]) and the solid
    #    (mapped z_s ∈ [0,1]) share this single Chebyshev basis.
    zcoord = d3.Coordinate('z')
    dist   = d3.Distributor(zcoord, dtype=np.complex128)
    zbasis = d3.Chebyshev(zcoord, size=N, bounds=(0, 1))
    # -- base-state velocity: U(z) = 1 − z  (Couette) --------------------
    z = dist.local_grid(zbasis)
    U = dist.Field(name='U', bases=zbasis)
    U['g'] = 1.0 - z
    # -- field variables --------------------------------------------------
    # Liquid fields (on physical z ∈ [0,1]):
    w        = dist.Field(name='w',        bases=zbasis)   # vertical velocity
    wz       = dist.Field(name='wz',       bases=zbasis)   # Dw
    eta      = dist.Field(name='eta',      bases=zbasis)   # z-vorticity = (D²−k²)w
    etaz     = dist.Field(name='etaz',     bases=zbasis)   # Dη
    theta_l  = dist.Field(name='theta_l',  bases=zbasis)   # liquid temperature
    theta_lz = dist.Field(name='theta_lz', bases=zbasis)   # Dθ_l
    # Solid fields (on mapped z_s ∈ [0,1]):
    theta_s  = dist.Field(name='theta_s',  bases=zbasis)   # solid temperature
    theta_sz = dist.Field(name='theta_sz', bases=zbasis)   # D_{z_s}θ_s
    # Scalar variable (interface perturbation amplitude):
    h_hat    = dist.Field(name='h_hat')                    # ĥ  (no basis → scalar)
    # Eigenvalue:
    sigma    = dist.Field(name='sigma')
    # -- tau fields (8 total: one per first-order equation) ----------------
    tau_w1   = dist.Field(name='tau_w1')
    tau_w2   = dist.Field(name='tau_w2')
    tau_eta1 = dist.Field(name='tau_eta1')
    tau_eta2 = dist.Field(name='tau_eta2')
    tau_tl1  = dist.Field(name='tau_tl1')
    tau_tl2  = dist.Field(name='tau_tl2')
    tau_ts1  = dist.Field(name='tau_ts1')
    tau_ts2  = dist.Field(name='tau_ts2')
    # -- operators --------------------------------------------------------
    dz   = lambda A: d3.Differentiate(A, zcoord)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    # -- build eigenvalue problem -----------------------------------------
    problem = d3.EVP(
        [w, wz, eta, etaz, theta_l, theta_lz,
         theta_s, theta_sz, h_hat,
         tau_w1, tau_w2, tau_eta1, tau_eta2,
         tau_tl1, tau_tl2, tau_ts1, tau_ts2],
        eigenvalue=sigma,
        namespace=locals()
    )
    # =====================================================================
    #  FIRST-ORDER REDUCTIONS  (4 equations)
    # =====================================================================
    # These define the auxiliary derivative variables.
    problem.add_equation("dz(w)       - wz       + lift(tau_w1,   -1) = 0")  # R1
    problem.add_equation("dz(eta)     - etaz     + lift(tau_eta1, -1) = 0")  # R2
    problem.add_equation("dz(theta_l) - theta_lz + lift(tau_tl1,  -1) = 0")  # R3
    problem.add_equation("dz(theta_s) - theta_sz + lift(tau_ts1,  -1) = 0")  # R4
    # =====================================================================
    #  GOVERNING EQUATIONS  (4 field equations + 1 scalar)
    # =====================================================================
    # -- E1a: Definition of z-vorticity -----------------------------------
    #    (D² − k²)w = η   →   dz(wz) − k²w − η = 0
    problem.add_equation(
        "dz(wz) - k2*w - eta + lift(tau_w2, -1) = 0"
    )
    # -- E1b: Orr-Sommerfeld (in terms of η) ------------------------------
    #    Pr(D² − k²)η + (σ − ikPeU)η − buoy·θ_l = 0
    #  → ση + Pr(dz(ηz) − k²η) − ikPeUη − buoy·θ_l = 0
    problem.add_equation(
        "sigma*eta + Pr*(dz(etaz) - k2*eta)"
        " - ik*Pe*U*eta - buoy*theta_l"
        " + lift(tau_eta2, -1) = 0"
    )
    # -- E2: Liquid energy ------------------------------------------------
    #    (D² − k²)θ_l + (σ − ikPeU)θ_l + Pe·w = 0
    #  → σθ_l + dz(θ_lz) − k²θ_l − ikPeUθ_l + Pe·w = 0
    problem.add_equation(
        "sigma*theta_l + dz(theta_lz) - k2*theta_l"
        " - ik*Pe*U*theta_l + Pe*w"
        " + lift(tau_tl2, -1) = 0"
    )
    # -- E3: Solid energy (mapped coordinate) -----------------------------
    #    Physical:  (D²_ζ − k² + σ)θ_s = 0
    #    Mapped:    (1/Λ²)D²_{z_s}θ_s − k²θ_s + σθ_s = 0
    #    ×Λ²:      D²_{z_s}θ_s − Λ²k²θ_s + Λ²σθ_s = 0
    #  → Λ²σθ_s + dz(θ_sz) − Λ²k²θ_s = 0
    problem.add_equation(
        "Lam2*sigma*theta_s + dz(theta_sz) - Lam2_k2*theta_s"
        " + lift(tau_ts2, -1) = 0"
    )
    # -- E4: Stefan condition (scalar equation for ĥ) ---------------------
    #    −σĥ = (1/ΛS)[D_ζ θ_s(ζ=1) − D_z θ_l(z=1)]
    #         = (1/ΛS)[(1/Λ)θ_sz(z_s=0) − θ_lz(z=1)]
    #  → −σĥ − (1/Λ²S)θ_sz(0) + (1/ΛS)θ_lz(1) = 0
    problem.add_equation(
        "-sigma*h_hat"
        " - coeff_s*theta_sz(z=0)"
        " + coeff_l*theta_lz(z=1) = 0"
    )
    # =====================================================================
    #  BOUNDARY CONDITIONS  (8 total = 8 taus)
    # =====================================================================
    # -- At z = 0: liquid bottom wall (rigid, hot, no-slip) ---------------
    problem.add_equation("w(z=0)       = 0")              # BC1: no penetration
    problem.add_equation("wz(z=0)      = 0")              # BC2: no slip
    problem.add_equation("theta_l(z=0) = 0")              # BC3: fixed temp θ_l=0
    #   (perturbation is zero because base θ_l(0)=1 is fixed)
    # -- At z = 1: liquid side of interface --------------------------------
    problem.add_equation("w(z=1)       = 0")              # BC4: no penetration
    problem.add_equation("wz(z=1) + ik*h_hat = 0")        # BC5: no-slip → Dw = −ikĥ
    problem.add_equation("theta_l(z=1) - h_hat = 0")      # BC6: melting temp → θ_l(1) = ĥ
    # -- At z_s = 0: solid side of interface (mapped ζ = 1) ----------------
    problem.add_equation("theta_s(z=0) - h_hat = 0")      # BC7: melting temp → θ_s(0) = ĥ
    # -- At z_s = 1: solid cold wall (mapped ζ = 1+Λ) ---------------------
    problem.add_equation("theta_s(z=1) = 0")              # BC8: fixed cold wall temp
    # =====================================================================
    #  SOLVE
    # =====================================================================
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return solver.eigenvalues, solver
# ======================================================================
# FILTERING UTILITIES
# ======================================================================
def filter_eigenvalues(evals, cutoff=1e6):
    """Remove spurious eigenvalues with very large magnitude."""
    good = np.isfinite(evals) & (np.abs(evals) < cutoff)
    return evals[good]
def filter_by_resolution_test(evals1, evals2, tolerance=1e-5):
    """
    Keep only the eigenvalues in evals1 that have a close match in evals2,
    indicating that they are physically converged and not spurious.
    """
    resolved = []
    for e1 in evals1:
        dists = np.abs(evals2 - e1)
        if len(dists) > 0:
            min_dist = np.min(dists)
            if min_dist < tolerance:
                resolved.append(e1)
    return np.array(resolved)
def get_most_unstable(Ra, Pe, Pr, S, Lam, k, N, tol):
    """
    Solve at resolution N and N+16, apply resolution test,
    and return the growth rate of the most unstable converged mode.
    """
    ev_N_raw, _ = solve_evp(Ra, Pe, Pr, S, Lam, k, N)
    ev_N = filter_eigenvalues(ev_N_raw)
    ev_hi_raw, _ = solve_evp(Ra, Pe, Pr, S, Lam, k, N + 16)
    ev_hi = filter_eigenvalues(ev_hi_raw)
    ev_res = filter_by_resolution_test(ev_N, ev_hi, tolerance=tol)
    if len(ev_res) > 0:
        sig = ev_res[np.argmin(ev_res.real)]
        return -sig.real                       # growth rate
    else:
        return np.nan
# ======================================================================
# MAIN: COMPUTE & PLOT  (MPI-parallel parameter sweep)
# ======================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    #  PARAMETERS
    # ------------------------------------------------------------------
    Pr   = 1.0
    S    = 1.0
    Lam  = 1.0
    Pe   = 10.0
    N    = 64        # Base Chebyshev modes
    tol  = 1e-4      # Convergence tolerance for resolution test
    # ------------------------------------------------------------------
    #  1)  EIGENVALUE SPECTRUM  for one (Ra, k) pair   [rank 0 only]
    # ------------------------------------------------------------------
    Ra_spec = 5000.0
    k_spec  = 3.0
    evals_for_plot = None
    if rank == 0:
        print("=" * 65)
        print(f"Phase-Boundary Stability — Dedalus v3  ({size} MPI rank(s))")
        print("=" * 65)
        print(f"\nParameters:  Pr={Pr}, S={S}, Λ={Lam}, Pe={Pe}, N={N}")
        print(f"Resolution-test tolerance: {tol}")
        print(f"\n--- Spectrum:  Ra={Ra_spec}, k={k_spec} ---")
        evals_N_raw, solver = solve_evp(Ra_spec, Pe, Pr, S, Lam, k_spec, N)
        evals_N = filter_eigenvalues(evals_N_raw)
        evals_hi_raw, _ = solve_evp(Ra_spec, Pe, Pr, S, Lam, k_spec, N + 16)
        evals_hi = filter_eigenvalues(evals_hi_raw)
        evals_for_plot = filter_by_resolution_test(evals_N, evals_hi, tolerance=tol)
        print(f"  Raw (magnitude-filtered): {len(evals_N)}  →  "
              f"Converged (resolution test): {len(evals_for_plot)}")
        if len(evals_for_plot) > 0:
            mu = evals_for_plot[np.argmin(evals_for_plot.real)]
            print(f"  Most unstable σ = {mu:.6f}")
            print(f"  Growth rate −Re(σ) = {-mu.real:.6f}")
            print("  ⇒  UNSTABLE" if mu.real < 0 else "  ⇒  Stable")
        else:
            print("  No converged eigenvalues found.")
    CW.Barrier()
    # ------------------------------------------------------------------
    #  2)  GROWTH RATE vs WAVENUMBER  –  MPI-parallel over k
    # ------------------------------------------------------------------
    k_vals  = np.linspace(0.5, 8.0, 30)
    Ra_list = [1000, 3000, 5000, 10000]
    colors  = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    if rank == 0:
        print("\n--- Growth-rate curves (parallel sweep) ---")
    # Each rank takes a round-robin slice of k_vals
    my_indices = np.arange(rank, len(k_vals), size)
    my_k_vals  = k_vals[my_indices]
    growth_full = {}          # will hold the reassembled arrays on rank 0
    for Ra_val in Ra_list:
        # local computation
        my_rates = np.array([
            get_most_unstable(Ra_val, Pe, Pr, S, Lam, kv, N, tol)
            for kv in my_k_vals
        ])
        # gather to rank 0
        all_rates = CW.gather(my_rates, root=0)
        if rank == 0:
            full = np.empty(len(k_vals))
            for r in range(size):
                idx = np.arange(r, len(k_vals), size)
                full[idx] = all_rates[r]
            growth_full[Ra_val] = full
            mx = np.nanmax(full) if not np.all(np.isnan(full)) else np.nan
            print(f"  Ra = {Ra_val:>6.0f}  done  (max growth rate = {mx:.4f})")
    # ------------------------------------------------------------------
    #  3)  FIGURE — two panels  [rank 0 only]
    # ------------------------------------------------------------------
    if rank == 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
        # ---- Panel (a): Eigenvalue spectrum ----
        ax = axes[0]
        if evals_for_plot is not None and len(evals_for_plot) > 0:
            stable   = evals_for_plot[evals_for_plot.real >= 0]
            unstable = evals_for_plot[evals_for_plot.real <  0]
            ax.scatter(stable.real, stable.imag, s=20, c='steelblue',
                       alpha=0.6, edgecolors='navy', linewidths=0.4,
                       label=f'Stable ({len(stable)})')
            if len(unstable) > 0:
                ax.scatter(unstable.real, unstable.imag, s=120, c='crimson',
                           marker='*', zorder=5, edgecolors='darkred',
                           linewidths=0.4,
                           label=f'Unstable ({len(unstable)})')
        ax.axvline(0, color='grey', ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel(r'$\mathrm{Re}(\sigma)$', fontsize=14)
        ax.set_ylabel(r'$\mathrm{Im}(\sigma)$', fontsize=14)
        ax.set_title(
            f'(a) Eigenvalue Spectrum (converged modes)\n'
            f'Ra={Ra_spec:.0f}, Pe={Pe}, k={k_spec}, Pr={Pr}, S={S}, '
            r'$\Lambda$' + f'={Lam}', fontsize=12)
        ax.legend(fontsize=11, loc='upper right')
        # ---- Panel (b): Growth rate vs wavenumber ----
        ax2 = axes[1]
        for Ra_val, color in zip(Ra_list, colors):
            ax2.plot(k_vals, growth_full[Ra_val], '-o', color=color,
                     markersize=3, lw=1.8, label=f'Ra = {Ra_val:.0f}')
        ax2.axhline(0, color='grey', ls='--', lw=0.8, alpha=0.6)
        ax2.set_xlabel(r'Wavenumber $k$', fontsize=14)
        ax2.set_ylabel(r'Growth rate $-\mathrm{Re}(\sigma)$', fontsize=14)
        ax2.set_title(
            f'(b) Growth Rate vs Wavenumber\n'
            f'Pe={Pe}, Pr={Pr}, S={S}, '
            r'$\Lambda$' + f'={Lam}', fontsize=12)
        ax2.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig('stability_spectrum.png', dpi=200, bbox_inches='tight')
        print(f"\nSaved figure → stability_spectrum.png")