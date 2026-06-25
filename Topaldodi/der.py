"""
===========================================================================
Eigenvalue Solver: Shear + Buoyancy + Phase Boundary Stability
===========================================================================
Solves the coupled Orr-Sommerfeld / Energy eigenvalue problem with
a Stefan (phase-change) boundary condition using Dedalus v3.

Reference: Toppaladoddi & Wettlaufer, J. Fluid Mech. (2019)
           (extended to keep full time-dependence in the coupled w-θ system)

Convention:  perturbations ~ exp(i k x - σ t)
             Re(σ) > 0  →  perturbation decays  (STABLE)
             Re(σ) < 0  →  perturbation grows    (UNSTABLE)

System (on z ∈ [0, 1]):
  OS:     σ p + Pr(D²−γ²)p − ikPeU p − γ² Ra Pr / Pe · θ = 0
  Defn:   (D²−γ²)w = p
  Energy: σ θ + (D²−γ²)θ − ikPeU θ + Pe w = 0

  where  p = (D²−γ²)w  is the "vorticity",  U(z) = 1−z  (Couette)

BCs:
  z = 0:  w = 0,  Dw = 0,  θ = 0       (rigid hot wall)
  z = 1:  w = 0,  Dw = −ik θ(1)         (no-slip on perturbed interface)
          Dθ(1) + γ coth(γΛ) θ(1) − σ Λ S θ(1) = 0   (Stefan / Robin)

Solid is solved analytically (quasi-steady: μ = γ).
===========================================================================
"""

import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)


# =========================================================================
# SOLVER
# =========================================================================

def solve_evp(Ra, Pe, Pr, S, Lam, k, N=64):
    """
    Solve the eigenvalue problem for given parameters.

    Parameters
    ----------
    Ra  : float – Rayleigh number
    Pe  : float – Peclet number (> 0)
    Pr  : float – Prandtl number
    S   : float – Stefan number
    Lam : float – Lambda = (T_m - T_c) / (T_h - T_m)
    k   : float – streamwise wavenumber (m = 0 for 2-D)
    N   : int   – Chebyshev resolution

    Returns
    -------
    eigenvalues : 1-D complex array
    """

    gamma2 = k**2
    gamma_val = abs(k)

    # -- coordinate & basis -------------------------------------------
    zcoord = d3.Coordinate('z')
    dist   = d3.Distributor(zcoord, dtype=np.complex128)
    zbasis = d3.Chebyshev(zcoord, size=N, bounds=(0, 1))

    # -- fields -------------------------------------------------------
    w      = dist.Field(name='w',      bases=zbasis)
    wz     = dist.Field(name='wz',     bases=zbasis)
    p      = dist.Field(name='p',      bases=zbasis)   # (D^2 - gamma^2) w
    pz     = dist.Field(name='pz',     bases=zbasis)
    theta  = dist.Field(name='theta',  bases=zbasis)
    thetaz = dist.Field(name='thetaz', bases=zbasis)
    sigma  = dist.Field(name='sigma')

    # base-state velocity  U(z) = 1 - z
    z = dist.local_grid(zbasis)
    U = dist.Field(name='U', bases=zbasis)
    U['g'] = 1.0 - z

    # -- tau fields (6 total: 2 per 2nd-order equation) ---------------
    tau_w1  = dist.Field(name='tau_w1')
    tau_w2  = dist.Field(name='tau_w2')
    tau_p1  = dist.Field(name='tau_p1')
    tau_p2  = dist.Field(name='tau_p2')
    tau_th1 = dist.Field(name='tau_th1')
    tau_th2 = dist.Field(name='tau_th2')

    # -- operators ----------------------------------------------------
    dz = lambda A: d3.Differentiate(A, zcoord)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A, n: d3.Lift(A, lift_basis, n)

    # -- solid gradient (quasi-steady: mu = gamma) --------------------
    if gamma_val * Lam > 1e-10:
        gcoth = gamma_val / np.tanh(gamma_val * Lam)
    else:
        gcoth = 1.0 / Lam                 # limiting form

    buoy = gamma2 * Ra * Pr / Pe           # buoyancy coupling coefficient

    # -- build eigenvalue problem -------------------------------------
    problem = d3.EVP(
        [w, wz, p, pz, theta, thetaz,
         tau_w1, tau_w2, tau_p1, tau_p2, tau_th1, tau_th2],
        eigenvalue=sigma,
        namespace=locals()
    )

    # first-order reductions
    problem.add_equation("dz(w)     - wz     + lift(tau_w1,  -1) = 0")
    problem.add_equation("dz(p)     - pz     + lift(tau_p1,  -1) = 0")
    problem.add_equation("dz(theta) - thetaz + lift(tau_th1, -1) = 0")

    # definition:  p = (D^2 - gamma^2) w
    problem.add_equation(
        "dz(wz) - gamma2*w - p + lift(tau_w2, -1) = 0"
    )

    # Orr-Sommerfeld:
    #   sigma * p  +  Pr (D^2 - gamma^2) p  -  ik Pe U p  -  buoy * theta = 0
    problem.add_equation(
        "sigma*p + Pr*(dz(pz) - gamma2*p) - 1j*k*Pe*U*p "
        "- buoy*theta + lift(tau_p2, -1) = 0"
    )

    # Energy:
    #   sigma * theta  +  (D^2 - gamma^2) theta  -  ik Pe U theta  +  Pe w = 0
    problem.add_equation(
        "sigma*theta + dz(thetaz) - gamma2*theta "
        "- 1j*k*Pe*U*theta + Pe*w + lift(tau_th2, -1) = 0"
    )

    # -- boundary conditions (6) --------------------------------------
    # z = 0  (rigid hot wall)
    problem.add_equation("w(z=0)     = 0")              # no penetration
    problem.add_equation("wz(z=0)    = 0")              # no slip
    problem.add_equation("theta(z=0) = 0")              # fixed temperature

    # z = 1  (solid-liquid interface)
    problem.add_equation("w(z=1) = 0")                  # no penetration
    problem.add_equation(                               # no slip: Dw = -ik theta(1)
        "wz(z=1) + 1j*k*theta(z=1) = 0"
    )
    problem.add_equation(                               # Stefan / Robin BC
        "-sigma*Lam*S*theta(z=1) + thetaz(z=1) + gcoth*theta(z=1) = 0"
    )

    # -- solve --------------------------------------------------------
    solver = problem.build_solver()
    
    # In Dedalus v3, the matrices for the 1D EVP are in the first subproblem
    subproblem = solver.subproblems[0]
    
    # Extract the sparse matrices (L and M)
    L_sparse = subproblem.L_ext
    M_sparse = subproblem.M_ext
    
    # Convert to dense numpy arrays so you can print or inspect them
    L_dense = L_sparse.toarray()
    M_dense = M_sparse.toarray()
    
    # Print the shapes (it will be an (8N+9) x (8N+9) matrix)
    print(f"L matrix shape: {L_dense.shape}")
    print(f"M matrix shape: {M_dense.shape}")
    
    # Plot the "sparsity pattern" (shows exactly where the non-zero elements are)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # spy() plots a dot for every non-zero element in the matrix
    ax1.spy(L_dense, markersize=1, color='navy')
    ax1.set_title("L Matrix (Spatial Physics & BCs)")
    
    ax2.spy(M_dense, markersize=1, color='crimson')
    ax2.set_title("M Matrix (Time Derivative / $\sigma$)")
    
    plt.tight_layout()
    plt.savefig('matrix_structure.png', dpi=200)
    plt.show()
    
    # You can also print the full matrices to the console or save to a text file:
    # np.savetxt("L_matrix.csv", np.real(L_dense), delimiter=",")
    # np.savetxt("M_matrix.csv", np.real(M_dense), delimiter=",")
    
    # Now solve the dense eigenvalue problem
    solver.solve_dense(subproblem)
    
    return solver.eigenvalues


def filter_eigenvalues(evals, cutoff=1e5):
    """Remove spurious eigenvalues with very large magnitude."""
    good = (np.isfinite(evals)) & (np.abs(evals) < cutoff)
    return evals[good]


def most_unstable(evals):
    """
    Return the most unstable eigenvalue.
    With e^{-sigma t}: most unstable = most negative Re(sigma).
    Growth rate = -Re(sigma);  positive means growing.
    """
    if len(evals) == 0:
        return np.nan + 0j
    return evals[np.argmin(evals.real)]


# =========================================================================
# MAIN: compute & plot
# =========================================================================

if __name__ == "__main__":

    # -- parameters ---------------------------------------------------
    Pr  = 1.0
    S   = 1.0
    Lam = 1.0
    Pe  = 10.0
    N   = 64        # Chebyshev modes

    # =================================================================
    # PLOT 1:  Eigenvalue spectrum  (sigma_r  vs  sigma_i)
    # =================================================================
    Ra_spec = 5000.0
    k_spec  = 3.0

    print(f"Computing spectrum: Ra={Ra_spec}, Pe={Pe}, k={k_spec} ...")
    evals_raw = solve_evp(Ra_spec, Pe, Pr, S, Lam, k_spec, N)
    evals     = filter_eigenvalues(evals_raw)

    mu = most_unstable(evals)
    print(f"  Most unstable eigenvalue: sigma = {mu:.6f}")
    print(f"  Growth rate -Re(sigma)         = {-mu.real:.6f}")

    # =================================================================
    # PLOT 2:  Growth rate  vs  wavenumber  for several Ra
    # =================================================================
    k_vals  = np.linspace(0.5, 8.0, 30)
    Ra_list = [1000, 3000, 5000, 10000]
    colors  = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    print("\nComputing growth-rate curves ...")
    growth_data = {}
    for Ra_val in Ra_list:
        rates = []
        for kv in k_vals:
            ev = filter_eigenvalues(solve_evp(Ra_val, Pe, Pr, S, Lam, kv, N))
            sig = most_unstable(ev)
            rates.append(-sig.real if np.isfinite(sig) else np.nan)
        growth_data[Ra_val] = np.array(rates)
        print(f"  Ra = {Ra_val:>6.0f}  done  "
              f"(max growth rate = {np.nanmax(rates):.4f})")

    # =================================================================
    # FIGURE
    # =================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # -- Panel (a): spectrum ------------------------------------------
    ax = axes[0]
   

    ax.scatter(evals.real , evals.imag, s=18, c='steelblue',
               alpha=0.6, edgecolors='navy', linewidths=0.4,
               label=f'Stable ({len(stable)})')
   

    ax.axvline(0, color='grey', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel(r'$\mathrm{Re}(\sigma)$', fontsize=14)
    ax.set_ylabel(r'$\mathrm{Im}(\sigma)$', fontsize=14)
    ax.set_title(
        f'(a) Eigenvalue spectrum\n'
        f'Ra={Ra_spec:.0f}, Pe={Pe}, k={k_spec}, Pr={Pr}, S={S}, '
        r'$\Lambda$' + f'={Lam}',
        fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(-60, 200)

    # -- Panel (b): growth rate vs k ----------------------------------
    ax2 = axes[1]
    for Ra_val, color in zip(Ra_list, colors):
        ax2.plot(k_vals, -growth_data[Ra_val], '-o', color=color,
                 markersize=3, lw=1.8, label=f'Ra = {Ra_val:.0f}')

    ax2.axhline(0, color='grey', ls='--', lw=0.8, alpha=0.6)
    ax2.set_xlabel('Wavenumber  $k$', fontsize=14)
    ax2.set_ylabel(r'Growth rate  $-\mathrm{Re}(\sigma)$', fontsize=14)
    ax2.set_title(
        f'(b) Growth rate vs wavenumber\n'
        f'Pe={Pe}, Pr={Pr}, S={S}, '
        r'$\Lambda$' + f'={Lam}',
        fontsize=12)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig('stability_spectrum.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("\nSaved figure -> stability_spectrum.png")
  