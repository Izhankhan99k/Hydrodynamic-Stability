import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  Physical parameters (adjust as needed)
# ═══════════════════════════════════════════════════════════
Ra   = 0.0          # Rayleigh number (set >0 for buoyancy)
Pe   = 4000.0      # Péclet number
Pr   =  1.0          # Prandtl number
S    = 1.0          # Stefan number
Lam  = 0.5          # Lambda = (Tm - Tc) / DeltaT
k    = 1.0          # streamwise wavenumber
m    = 0.0          # spanwise wavenumber (set to 0 for 2D)
gamma2 = k**2 + m**2
d0     = Lam        # base-state solid thickness (d0 = Lambda)

# ═══════════════════════════════════════════════════════════
#  EVP solver function
# ═══════════════════════════════════════════════════════════
def solve_evp(N):
    """
    Build and solve the coupled generalised eigenvalue problem
    at Chebyshev resolution N. Returns the array of eigenvalues.
    """
    # ── coordinate, basis ──────────────────────────────────
    coord = d3.Coordinate('z')
    dist  = d3.Distributor(coord, dtype=np.complex128)
    basis = d3.Chebyshev(coord, size=N, bounds=(0, 1))
    z     = dist.local_grid(basis)

    # ── operators ──────────────────────────────────────────
    dz = lambda A: d3.Differentiate(A, coord)
    try:
        lift_basis = basis.derivative_basis(1)
    except AttributeError:
        lift_basis = basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    # ── fields ─────────────────────────────────────────────
    w   = dist.Field(name='w',   bases=basis)   # vertical velocity
    wz  = dist.Field(name='wz',  bases=basis)   # D w
    Lw  = dist.Field(name='Lw',  bases=basis)   # (D^2 - γ²) w
    Lwz = dist.Field(name='Lwz', bases=basis)   # D (Lw)
    tl  = dist.Field(name='tl',  bases=basis)   # liquid temperature perturbation
    tlz = dist.Field(name='tlz', bases=basis)   # D tl
    ts  = dist.Field(name='ts',  bases=basis)   # solid temperature perturbation
    tsz = dist.Field(name='tsz', bases=basis)   # D ts

    h     = dist.Field(name='h')                # interface amplitude
    sigma = dist.Field(name='sigma')            # eigenvalue (complex)

    # tau fields (8 BCs)
    tw1  = dist.Field(name='tw1')
    tw2  = dist.Field(name='tw2')
    tw3  = dist.Field(name='tw3')
    tw4  = dist.Field(name='tw4')
    ttl1 = dist.Field(name='ttl1')
    ttl2 = dist.Field(name='ttl2')
    tts1 = dist.Field(name='tts1')
    tts2 = dist.Field(name='tts2')

    # ── base-state velocity: POISEUILLE profile ────────────
    U0 = dist.Field(name='U0', bases=basis)
    U0['g'] = 4.0 * z * (1.0 - z)          # parabolic, max=1 at centre

    # ── namespace ──────────────────────────────────────────
    ns = dict(
        Ra=Ra, Pe=Pe, Pr=Pr, S=S, Lam=Lam,
        k=k, gamma2=gamma2, d0=d0,
        w=w, wz=wz, Lw=Lw, Lwz=Lwz,
        tl=tl, tlz=tlz, ts=ts, tsz=tsz,
        h=h, sigma=sigma, U0=U0,
        dz=dz, lift=lift,
        tw1=tw1, tw2=tw2, tw3=tw3, tw4=tw4,
        ttl1=ttl1, ttl2=ttl2, tts1=tts1, tts2=tts2,
    )

    variables = [w, wz, Lw, Lwz, tl, tlz, ts, tsz, h,
                 tw1, tw2, tw3, tw4, ttl1, ttl2, tts1, tts2]
    problem = d3.EVP(variables, eigenvalue=sigma, namespace=ns)

    # ═══════════════════════════════════════════════════════
    #  ODEs (first-order system with tau lifting)
    # ═══════════════════════════════════════════════════════

    # ── Orr‑Sommerfeld (4th order in w) ──────────────────
    #   Pr (D²-γ²)² w  -  ikPe U0 (D²-γ²) w  +  ikPe U'' w
    #   - γ² RaPr/Pe θ_l  +  iσ (D²-γ²) w = 0
    #   with U'' = -8  →  ikPe U'' = -8 ikPe
    problem.add_equation("wz  - dz(w)                  + lift(tw1)  = 0")
    problem.add_equation("dz(wz) - gamma2*w - Lw       + lift(tw2)  = 0")
    problem.add_equation("Lwz - dz(Lw)                 + lift(tw3)  = 0")
    problem.add_equation(
        "Pr*(dz(Lwz) - gamma2*Lw)"              # Pr (D²-γ²) Lw
        " - 1j*k*Pe*U0*Lw"                      # -ikPe U0 Lw
        " - 8*1j*k*Pe*w"                        # curvature term: +ikPe U'' w (U''=-8)
        " - gamma2*Ra*Pr/Pe*tl"                 # -γ² RaPr/Pe θ_l
        " + 1j*sigma*Lw"                        # + iσ Lw
        " + lift(tw4) = 0"
    )

    # ── Liquid heat (2nd order) ──────────────────────────
    #   (D²-γ²)θ_l - ikPe U0 θ_l + Pe w + iσ θ_l = 0
    problem.add_equation("tlz - dz(tl) + lift(ttl1) = 0")
    problem.add_equation(
        "dz(tlz) - gamma2*tl"                   # (D²-γ²) θ_l
        " - 1j*k*Pe*U0*tl"                      # -ikPe U0 θ_l
        " + Pe*w"                               # +Pe w  (from base gradient -1)
        " + 1j*sigma*tl"                        # +iσ θ_l
        " + lift(ttl2) = 0"
    )

    # ── Solid heat (2nd order, mapped domain) ────────────
    #   (1/d0²) D² θ_s - γ² θ_s + iσ θ_s = 0
    problem.add_equation("tsz - dz(ts) + lift(tts1) = 0")
    problem.add_equation(
        "1/d0**2*dz(tsz)"                       # (1/d0²) D² θ_s
        " - gamma2*ts"                          # -γ² θ_s
        " + 1j*sigma*ts"                        # +iσ θ_s
        " + lift(tts2) = 0"
    )

    # ═══════════════════════════════════════════════════════
    #  Boundary Conditions (8 BCs)
    # ═══════════════════════════════════════════════════════
    # z = 0 : bottom wall
    problem.add_equation("w(z=0)  = 0")          # no penetration
    problem.add_equation("wz(z=0) = 0")          # no slip (U0=0)
    problem.add_equation("tl(z=0) = 0")          # fixed hot temperature (perturbation 0)
    problem.add_equation("ts(z=0) = 0")          # fixed cold temperature (solid top, perturbation 0)

    # z = 1 : interface
    problem.add_equation("w(z=1)  = 0")          # kinematic BC
    # no‑slip: Dw(1) + U0'(1) * (i k h) = 0  with U0'(1) = -4
    problem.add_equation("wz(z=1) + 4*1j*k*h = 0")   # hence Dw(1) + 4 i k h = 0
    problem.add_equation("tl(z=1) - h = 0")          # melting temperature, liquid
    problem.add_equation("ts(z=1) - h = 0")          # melting temperature, solid

    # ═══════════════════════════════════════════════════════
    #  Stefan Condition (scalar equation for h)
    # ═══════════════════════════════════════════════════════
    #   - (1/d0) tsz(1) - tlz(1) + iσ Lam S h = 0
    problem.add_equation(
        "+1/d0*tsz(z=1) - tlz(z=1) + 1j*sigma*Lam*S*h = 0"
    )

    # ── solve ──────────────────────────────────────────────
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)

# ═══════════════════════════════════════════════════════════
#  Solve at two resolutions for convergence filtering
# ═══════════════════════════════════════════════════════════
N1, N2 = 200, 250
logger.info(f"Solving EVP at N = {N1} ...")
ev1 = solve_evp(N1)
logger.info(f"  → {len(ev1)} raw eigenvalues")
logger.info(f"Solving EVP at N = {N2} for convergence check ...")
ev2 = solve_evp(N2)
logger.info(f"  → {len(ev2)} raw eigenvalues")

# ── convergence filter ────────────────────────────────────
def convergence_filter(ev_lo, ev_hi, tol=0.05):
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e):
            continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    if len(good) == 0:
        logger.warning("Convergence filter returned nothing → "
                       "falling back to magnitude filter |σ| < 10·Pe")
        mask = np.isfinite(ev_lo) & (np.abs(ev_lo) < 10 * Pe)
        return ev_lo[mask]
    return ev_lo[good]

evals = convergence_filter(ev1, ev2, tol=0.05)
logger.info(f"Retained {len(evals)} / {len(ev1)} converged eigenvalues")

cr = evals.real / (k * Pe)      # phase speed   c_r = Re(σ)/(k Pe)
ci = evals.imag / (k * Pe)      # growth rate   c_i = Im(σ)/(k Pe)

idx = np.argmax(ci)
logger.info(f"Most unstable mode:  σ = {cr[idx]:.4f} {ci[idx]:+.4f}i")

# ═══════════════════════════════════════════════════════════
#  Plot the spectrum
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(cr, ci, s=55, marker='o',
           facecolors='none', edgecolors='blue',
           linewidths=1.2, zorder=3, label='Eigenmodes')
ax.scatter(cr[idx], ci[idx], s=250, marker='*',
           color='gold', edgecolors='k',
           linewidths=0.8, zorder=5, label='Most Unstable Mode')
ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.7)
ax.set_xlabel(r'Phase Speed  ($c_r$)', fontsize=14)
ax.set_ylabel(r'Growth Rate  ($c_i$)', fontsize=14)
ax.set_title(
    f'Poiseuille Flow with Phase Change\n'
    f'Ra={Ra:.0f},  Pe={Pe:.1f},  k={k}',
    fontsize=14)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, ls=':', alpha=0.5)
ax.tick_params(labelsize=12)
# Adjust plot limits as needed; you may want to zoom in.
plt.xlim(-0.25, 1)
plt.ylim(-3, 1)
plt.tight_layout()
plt.savefig('eigenvalue_spectrum_Poiseuille.png', dpi=500)
plt.show()

logger.info("Plot saved → eigenvalue_spectrum_Poiseuille.png")