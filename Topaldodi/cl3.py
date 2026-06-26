import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ═══════════════════════════════════════════════════════════
#  Physical parameters
# ═══════════════════════════════════════════════════════════
Ra   = 0     # Rayleigh number
Pe   = 40000.0      # Peclet number
Pr   = 7.0         # Prandtl number
S    = .1        # Stefan number
Lam  = 0.5         # Lambda = (Tm - Tc) / DeltaT
k    = 1.0         # streamwise wavenumber
m    = 0.0         # spanwise wavenumber
gamma2 = k**2 + m**2
d0     = Lam        # base-state solid thickness  (d0 = Lambda)
# ═══════════════════════════════════════════════════════════
#  EVP solver function
# ═══════════════════════════════════════════════════════════
def solve_evp(N):
    """
    Build and solve the coupled generalised eigenvalue problem
    at Chebyshev resolution N.  Returns the array of eigenvalues.
    """
    # ── coordinate, basis ──────────────────────────────────
    coord = d3.Coordinate('z')
    dist  = d3.Distributor(coord, dtype=np.complex128)
    basis = d3.Chebyshev(coord, size=N, bounds=(0, 1))
    z     = dist.local_grid(basis)
    # ── operators ──────────────────────────────────────────
    dz = lambda A: d3.Differentiate(A, coord)
    # lift operator for tau method (proper Dedalus 3 way)
    try:
        lift_basis = basis.derivative_basis(1)
    except AttributeError:
        lift_basis = basis          # fallback for older builds
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    # ── basis-dependent fields ─────────────────────────────
    w   = dist.Field(name='w',   bases=basis)   # vertical velocity
    wz  = dist.Field(name='wz',  bases=basis)   # Dw
    Lw  = dist.Field(name='Lw',  bases=basis)   # (D^2 - g^2)w
    Lwz = dist.Field(name='Lwz', bases=basis)   # D[(D^2 - g^2)w]
    tl  = dist.Field(name='tl',  bases=basis)   # theta_l
    tlz = dist.Field(name='tlz', bases=basis)   # D(theta_l)
    ts  = dist.Field(name='ts',  bases=basis)   # theta_s
    tsz = dist.Field(name='tsz', bases=basis)   # D_zeta(theta_s)
    # ── scalar fields ──────────────────────────────────────
    h     = dist.Field(name='h')                # interface amplitude
    sigma = dist.Field(name='sigma')            # eigenvalue
    # ── tau fields  (8 total → one per BC) ─────────────────
    tw1  = dist.Field(name='tw1')
    tw2  = dist.Field(name='tw2')
    tw3  = dist.Field(name='tw3')
    tw4  = dist.Field(name='tw4')
    ttl1 = dist.Field(name='ttl1')
    ttl2 = dist.Field(name='ttl2')
    tts1 = dist.Field(name='tts1')
    tts2 = dist.Field(name='tts2')
    # ── base-state profile  u^(0)(z) = 1 - z ──────────────
    U0 = dist.Field(name='U0', bases=basis)
    U0['g'] = 1.0 - z
    # ── namespace for equation parser ──────────────────────
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
    #  ODEs  (first-order system, lift(tau) for tau method)
    # ═══════════════════════════════════════════════════════
    # ── Orr-Sommerfeld  (4th order in w) ──────────────────
    #
    #  Pr(D^2-g^2)^2 w  -  ikPe u0 (D^2-g^2) w
    #       - g^2 RaPr/Pe  tl  +  is (D^2-g^2) w  =  0
    #
    # Auxiliary:  wz = Dw,  Lw = Dwz - g^2 w,  Lwz = DLw
    problem.add_equation("wz  - dz(w)                  + lift(tw1)  = 0")
    problem.add_equation("dz(wz) - gamma2*w - Lw       + lift(tw2)  = 0")
    problem.add_equation("Lwz - dz(Lw)                 + lift(tw3)  = 0")
    problem.add_equation(
        "Pr*(dz(Lwz) - gamma2*Lw)"           # Pr (D^2-g^2) Lw
        " - 1j*k*Pe*U0*Lw"                   # -ikPe u0 Lw
        " - gamma2*Ra*Pr/Pe*tl"              # -g^2 RaPr/Pe theta_l
        " + 1j*sigma*Lw"                     # +is Lw   (eigenvalue)
        " + lift(tw4) = 0"
    )
    # ── Liquid heat  (2nd order) ──────────────────────────
    #
    #  (D^2-g^2) tl  -  ikPe u0 tl  +  Pe w  +  is tl  =  0
    #
    #  The  "+Pe w" sign  comes from  D theta_l^(0) = -1:
    #    the advection term  Pe w' (-1)  is on the PDE's LHS,
    #    flipping sign when moved to the operator side.
    problem.add_equation("tlz - dz(tl) + lift(ttl1) = 0")
    problem.add_equation(
        "dz(tlz) - gamma2*tl"                # (D^2-g^2) tl
        " - 1j*k*Pe*U0*tl"                   # -ikPe u0 tl
        " + Pe*w"                             # +Pe w   (coupling)
        " + 1j*sigma*tl"                     # +is tl   (eigenvalue)
        " + lift(ttl2) = 0"
    )
    # ── Solid heat  (2nd order, mapped domain) ────────────
    #
    #  (1/d0^2) D_zeta^2 ts  -  g^2 ts  +  is ts  =  0
    problem.add_equation("tsz - dz(ts) + lift(tts1) = 0")
    problem.add_equation(
        "1/d0**2*dz(tsz)"                    # (1/d0^2) D^2 ts
        " - gamma2*ts"                        # -g^2 ts
        " + 1j*sigma*ts"                     # +is ts   (eigenvalue)
        " + lift(tts2) = 0"
    )
    # ═══════════════════════════════════════════════════════
    #  Boundary Conditions  (8 BCs absorb 8 tau fields)
    # ═══════════════════════════════════════════════════════
    # zeta = 0 :  liquid bottom wall  +  solid top wall
    problem.add_equation("w(z=0)  = 0")       # no penetration
    problem.add_equation("wz(z=0) = 0")       # no slip
    problem.add_equation("tl(z=0) = 0")       # fixed T_h
    problem.add_equation("ts(z=0) = 0")       # fixed T_c
    # zeta = 1 :  phase interface (for BOTH phases)
    problem.add_equation("w(z=1)  = 0")       # kinematic
    problem.add_equation("wz(z=1) + 1j*k*h = 0")   # no-slip  (Du0=-1)
    problem.add_equation("tl(z=1) - h = 0")         # melting T, liquid
    problem.add_equation("ts(z=1) - h = 0")         # melting T, solid
    # ═══════════════════════════════════════════════════════
    #  Stefan Condition  (governs scalar h-hat)
    # ═══════════════════════════════════════════════════════
    #
    #  [D_zs theta_s  -  D_zl theta_l]|_{z=1}  =  -is Lam S h
    #
    #  -iσ Λ S h_hat = - (-1/d0 D_zeta ts) + D_zeta tl
    #  0 = 1/d0*tsz(z=1) + tlz(z=1) + 1j*sigma*Lam*S*h
    problem.add_equation(
        "1/d0*tsz(z=1) + tlz(z=1) + 1j*sigma*Lam*S*h = 0"
    )
    # ── solve ──────────────────────────────────────────────
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)
# ═══════════════════════════════════════════════════════════
#  Solve at TWO resolutions for convergence-based filtering
# ═══════════════════════════════════════════════════════════
N1, N2 = 200,220
logger.info(f"Solving EVP at N = {N1} ...")
ev1 = solve_evp(N1)
logger.info(f"  → {len(ev1)} raw eigenvalues")
logger.info(f"Solving EVP at N = {N2} for convergence check ...")
ev2 = solve_evp(N2)
logger.info(f"  → {len(ev2)} raw eigenvalues")
# ═══════════════════════════════════════════════════════════
#  Convergence filter
#  Physical eigenvalues converge between resolutions.
#  Spurious tau-modes drift wildly → get removed.
# ═══════════════════════════════════════════════════════════
def convergence_filter(ev_lo, ev_hi, tol=0.05):
    """Keep eigenvalues from ev_lo that have a partner in ev_hi."""
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e):
            continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    # Base filter based on physical boundaries
    # Phase speed bounded around [-Pe, Pe] and Growth rate bounded c_i > -200
    phys_mask = np.isfinite(ev_lo) & (np.abs(ev_lo.real) < 2 * Pe) & (ev_lo.imag > -200)
    
    if len(good) == 0:
        logger.warning("Convergence filter returned nothing → falling back to physical bounds")
        return ev_lo[phys_mask]
        
    # Apply physical mask even to converged modes to remove highly damped diffusion modes
    converged = np.array(good)
    final_mask = phys_mask[converged]
    return ev_lo[converged[final_mask]]
evals = convergence_filter(ev1, ev2, tol=0.05)
logger.info(f"Retained {len(evals)} / {len(ev1)} converged eigenvalues")
cr = evals.real /(k*Pe)    # phase speed    Re(sigma)
ci = evals.imag /(k*Pe)    # growth rate    Im(sigma)
idx = np.argmax(ci)
logger.info(f"Most unstable mode:  σ = {cr[idx]:.4f} {ci[idx]:+.4f}i")
# ═══════════════════════════════════════════════════════════
#  Plot  (matching reference style)
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(cr, ci, s=55, marker='o',
           facecolors='none', edgecolors='blue',
           linewidths=1.2, zorder=3, label='Eigenmodes')
ax.scatter(cr[idx], ci[idx], s=250, marker='*',
           color='gold', edgecolors='k',
           linewidths=0.8, zorder=5, label='Most Unstable Mode')
ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.7)
# Zoom in to the physically relevant region
c_r_margin = 100
ax.set_xlim(-1000,1000)
ax.set_ylim(-5000,1000)
ax.set_xlabel(r'Phase Speed  ($c_r$)', fontsize=14)
ax.set_ylabel(r'Growth Rate  ($c_i$)',  fontsize=14)
ax.set_title(
    f'Coupled Phase-Change Spectrum\n'
    f'Ra={Ra:.0f},  Pe={Pe:.1f},  k={k}',
    fontsize=14)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, ls=':', alpha=0.5)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('eigenvalue_spectrum.png', dpi=150)
plt.show()
logger.info("Plot saved → eigenvalue_spectrum.png")
