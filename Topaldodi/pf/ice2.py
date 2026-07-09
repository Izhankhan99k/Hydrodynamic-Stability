import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ═══════════════════════════════════════════════════════════
#  Parameters (same non-dimensionalization as the paper)
# ═══════════════════════════════════════════════════════════
Pe = 5772.0       # Peclet number
Pr = 1.0           # Prandtl number  (Re = Pe/Pr = 10000)
k  = 1.0           # streamwise wavenumber
# ═══════════════════════════════════════════════════════════
#  EVP solver
# ═══════════════════════════════════════════════════════════
def solve_os(N):
    coord = d3.Coordinate('z')
    dist  = d3.Distributor(coord, dtype=np.complex128)
    basis = d3.Chebyshev(coord, size=N, bounds=(0, 1))
    z     = dist.local_grid(basis)
    dz = lambda A: d3.Differentiate(A, coord)
    try:
        lift_basis = basis.derivative_basis(1)
    except AttributeError:
        lift_basis = basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    # Fields
    w   = dist.Field(name='w',   bases=basis)
    wz  = dist.Field(name='wz',  bases=basis)
    Lw  = dist.Field(name='Lw',  bases=basis)   # (D^2 - k^2)w
    Lwz = dist.Field(name='Lwz', bases=basis)
    sigma = dist.Field(name='sigma')             # eigenvalue
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    tau3 = dist.Field(name='tau3')
    tau4 = dist.Field(name='tau4')
    # Base flow: Poiseuille
    U0 = dist.Field(name='U0', bases=basis)
    U0['g'] = 4.0 * z * (1.0 - z)
    U0zz = dist.Field(name='U0zz', bases=basis)
    U0zz['g'] = -8.0 * np.ones_like(z)
    ns = dict(
        Pe=Pe, Pr=Pr, k=k,
        w=w, wz=wz, Lw=Lw, Lwz=Lwz, sigma=sigma,
        tau1=tau1, tau2=tau2, tau3=tau3, tau4=tau4,
        dz=dz, lift=lift, U0=U0, U0zz=U0zz,
    )
    problem = d3.EVP([w, wz, Lw, Lwz, tau1, tau2, tau3, tau4],
                     eigenvalue=sigma, namespace=ns)
    # First-order reduction + Orr-Sommerfeld equation
    problem.add_equation("wz  - dz(w)                    + lift(tau1) = 0")
    problem.add_equation("dz(wz) - (k**2)*w - Lw         + lift(tau2) = 0")
    problem.add_equation("Lwz - dz(Lw)                   + lift(tau3) = 0")
    problem.add_equation(
        "Pr*(dz(Lwz) - (k**2)*Lw)"          # Pr (D^2-k^2)^2 w
        " - 1j*k*Pe*(U0*Lw - U0zz*w)"       # -ikPe (U0 Lw - U0'' w)
        " + 1j*sigma*Lw"                    # +i sigma Lw   (eigenvalue)
        " + lift(tau4) = 0"
    )
    # RIGID WALL boundary conditions (the key difference!)
    problem.add_equation("w(z=0)  = 0")      # no penetration, bottom
    problem.add_equation("wz(z=0) = 0")      # no slip, bottom
    problem.add_equation("w(z=1)  = 0")      # no penetration, top
    problem.add_equation("wz(z=1) = 0")      # no slip, top  ← RIGID WALL
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)
# ═══════════════════════════════════════════════════════════
#  Two-resolution convergence filter
# ═══════════════════════════════════════════════════════════
N1, N2 = 128, 192
logger.info(f"Solving OS at N = {N1} ...")
ev1 = solve_os(N1)
logger.info(f"  → {len(ev1)} raw eigenvalues")
logger.info(f"Solving OS at N = {N2} ...")
ev2 = solve_os(N2)
logger.info(f"  → {len(ev2)} raw eigenvalues")
def convergence_filter(ev_lo, ev_hi, tol=1e-6):
    """Keep eigenvalues from ev_lo that have a partner in ev_hi."""
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e):
            continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    if len(good) == 0:
        logger.warning("No converged modes found!")
        return ev_lo[np.isfinite(ev_lo)]
    return ev_lo[np.array(good)]
evals = convergence_filter(ev1, ev2, tol=1e-6)
# Convert sigma → c = sigma/k  (phase speed)
c_vals = evals / (k *Pe)
cr = c_vals.real
ci = c_vals.imag
logger.info(f"Retained {len(evals)} converged eigenvalues")
idx = np.argmax(ci)
logger.info(f"Most unstable mode: c = {cr[idx]:.6f} + {ci[idx]:.6f}i")
# ═══════════════════════════════════════════════════════════
#  Plot the classic Y-shaped / tree-shaped spectrum
# ═══════════════════════════════════════════════════════════
Re = Pe / Pr
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(cr, ci, s=40, marker='o',
           facecolors='none', edgecolors='royalblue',
           linewidths=1.0, zorder=3, label='Eigenmodes')
ax.scatter(cr[idx], ci[idx], s=250, marker='*',
           color='gold', edgecolors='k',
           linewidths=0.8, zorder=5, label='Most Unstable Mode')
ax.annotate(f"({cr[idx]:.4f}, {ci[idx]:.4f})",
            xy=(cr[idx], ci[idx]), xycoords='data',
            xytext=(15, 15), textcoords='offset points',
            fontsize=10, fontweight='bold', color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='gray'))
ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.7)
ax.set_xlim(-1,4)
ax.set_ylim(-10, 2)
ax.set_xlabel(r'Phase Speed  $c_r$', fontsize=14)
ax.set_ylabel(r'Growth Rate  $c_i$', fontsize=14)
ax.set_title(
    f'Plane Poiseuille Orr-Sommerfeld Spectrum (Rigid Walls)\n'
    f'Re = {Re:.0f},  k = {k}',
    fontsize=14)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, ls=':', alpha=0.5)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('poiseuille_os_tree.png', dpi=150)
plt.show()
logger.info("Plot saved → poiseuille_os_tree.png")
