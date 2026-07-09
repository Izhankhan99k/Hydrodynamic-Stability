
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  Physical parameters
# ═══════════════════════════════════════════════════════════
Ra   = 0   # Rayleigh number
Pe   = 40000  # Peclet number
Pr   = 1.0          # Prandtl number  (Re = Pe/Pr)
S    = 0.000001         # Stefan number
Lam  = 0.5         # Λ = (Tm − Tc) / ΔT
k    = 1         # streamwise wavenumber
m    = 0.0          # spanwise wavenumber
gamma2 = k**2 + m**2
d0     = Lam         # base-state solid thickness (from flux balance: d₀ = Λ)
# ═══════════════════════════════════════════════════════════
#  EVP solver
# ═══════════════════════════════════════════════════════════
def solve_evp(N):
    """
    Build and solve the coupled generalised eigenvalue problem
    at Chebyshev resolution N.  Returns the array of eigenvalues σ.
    """
    # ── coordinate & basis ────────────────────────────────
    coord = d3.Coordinate('z')
    dist  = d3.Distributor(coord, dtype=np.complex128)
    basis = d3.Chebyshev(coord, size=N, bounds=(0, 1))
    z     = dist.local_grid(basis)
    # ── operators ─────────────────────────────────────────
    dz = lambda A: d3.Differentiate(A, coord)
    try:
        lift_basis = basis.derivative_basis(1)
    except AttributeError:
        lift_basis = basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    # ── fields on Chebyshev basis ─────────────────────────
    w   = dist.Field(name='w',   bases=basis)   # vertical velocity
    wz  = dist.Field(name='wz',  bases=basis)   # Dw
    Lw  = dist.Field(name='Lw',  bases=basis)   # (D²−γ²)w
    Lwz = dist.Field(name='Lwz', bases=basis)   # D(Lw)
    tl  = dist.Field(name='tl',  bases=basis)   # liquid temperature pert.
    tlz = dist.Field(name='tlz', bases=basis)   # D(θ_l)
    ts  = dist.Field(name='ts',  bases=basis)   # solid temperature pert.
    tsz = dist.Field(name='tsz', bases=basis)   # D_ζ(θ_s)
    # ── scalar fields ─────────────────────────────────────
    h     = dist.Field(name='h')                # interface deformation
    sigma = dist.Field(name='sigma')            # eigenvalue
    # ── tau fields (8 BCs → 8 taus) ──────────────────────
    tw1  = dist.Field(name='tw1')
    tw2  = dist.Field(name='tw2')
    tw3  = dist.Field(name='tw3')
    tw4  = dist.Field(name='tw4')
    ttl1 = dist.Field(name='ttl1')
    ttl2 = dist.Field(name='ttl2')
    tts1 = dist.Field(name='tts1')
    tts2 = dist.Field(name='tts2')
    # ── Poiseuille base-state velocity ────────────────────
    #    U₀(z) = 4z(1−z),  U₀' = 4−8z,  U₀'' = −8
    U0 = dist.Field(name='U0', bases=basis)
    U0['g'] = 4.0 * z * (1.0 - z)
    U0z = dist.Field(name='U0z', bases=basis)
    U0z['g'] = 4.0 - 8.0 * z
    U0zz = dist.Field(name='U0zz', bases=basis)
    U0zz['g'] = -8.0 * np.ones_like(z)
    # ── namespace for equation parser ─────────────────────
    ns = dict(
        Ra=Ra, Pe=Pe, Pr=Pr, S=S, Lam=Lam,
        k=k, gamma2=gamma2, d0=d0,
        w=w, wz=wz, Lw=Lw, Lwz=Lwz,
        tl=tl, tlz=tlz, ts=ts, tsz=tsz,
        h=h, sigma=sigma,
        U0=U0, U0z=U0z, U0zz=U0zz,
        dz=dz, lift=lift,
        tw1=tw1, tw2=tw2, tw3=tw3, tw4=tw4,
        ttl1=ttl1, ttl2=ttl2, tts1=tts1, tts2=tts2,
    )
    variables = [w, wz, Lw, Lwz, tl, tlz, ts, tsz, h,
                 tw1, tw2, tw3, tw4, ttl1, ttl2, tts1, tts2]
    problem = d3.EVP(variables, eigenvalue=sigma, namespace=ns)
    # ═══════════════════════════════════════════════════════
    #  PDEs (first-order reduction with tau method)
    # ═══════════════════════════════════════════════════════
    # ── Orr-Sommerfeld (4th order in w) ───────────────────
    #
    #  Pr(D²−γ²)²w − ikPe[U₀(D²−γ²)w − U₀''w]
    #       − γ² RaPr/Pe θ_l  +  iσ(D²−γ²)w  =  0
    #
    #  First-order auxiliaries:
    #    wz  = Dw
    #    Lw  = Dwz − γ²w   [= (D²−γ²)w]
    #    Lwz = D(Lw)
    problem.add_equation("wz  - dz(w)                  + lift(tw1) = 0")
    problem.add_equation("dz(wz) - gamma2*w - Lw       + lift(tw2) = 0")
    problem.add_equation("Lwz - dz(Lw)                 + lift(tw3) = 0")
    problem.add_equation(
        "Pr*(dz(Lwz) - gamma2*Lw)"            # Pr(D²−γ²)²w
        " - 1j*k*Pe*(U0*Lw - U0zz*w)"         # −ikPe(U₀·Lw − U₀''·w)
        " - gamma2*Ra*Pr/Pe*tl"                # −γ²·RaPr/Pe·θ_l
        " + 1j*sigma*Lw"                       # +iσ·Lw  (eigenvalue)
        " + lift(tw4) = 0"
    )
    # ── Liquid heat (2nd order) ───────────────────────────
    #
    #  (D²−γ²)θ_l − ikPe·U₀·θ_l + Pe·w + iσ·θ_l = 0
    #
    #  The +Pe·w comes from advecting the base gradient Dθ_l⁰ = −1:
    #    −Pe·w·(−1) = +Pe·w
    problem.add_equation("tlz - dz(tl) + lift(ttl1) = 0")
    problem.add_equation(
        "dz(tlz) - gamma2*tl"                  # (D²−γ²)θ_l
        " - 1j*k*Pe*U0*tl"                     # −ikPe·U₀·θ_l
        " + Pe*w"                               # +Pe·w
        " + 1j*sigma*tl"                        # +iσ·θ_l  (eigenvalue)
        " + lift(ttl2) = 0"
    )
    # ── Solid heat (2nd order, mapped domain) ─────────────
    #
    #  (1/d₀²)D_ζ²θ_s − γ²θ_s + iσ·θ_s = 0
    problem.add_equation("tsz - dz(ts) + lift(tts1) = 0")
    problem.add_equation(
        "1/d0**2*dz(tsz)"                      # (1/d₀²)D²θ_s
        " - gamma2*ts"                          # −γ²θ_s
        " + 1j*sigma*ts"                        # +iσ·θ_s  (eigenvalue)
        " + lift(tts2) = 0"
    )
    # ═══════════════════════════════════════════════════════
    #  Boundary Conditions (8 BCs absorb 8 tau fields)
    # ═══════════════════════════════════════════════════════
    # z = 0 : bottom wall (liquid) + far cold wall (solid, ζ=0)
    problem.add_equation("w(z=0)  = 0")            # no penetration
    problem.add_equation("wz(z=0) = 0")            # no slip
    problem.add_equation("tl(z=0) = 0")            # fixed T_h perturbation
    problem.add_equation("ts(z=0) = 0")            # fixed T_c perturbation
    # z = 1 : melting interface (ζ=1 for both liquid and solid)
    problem.add_equation("w(z=1)  = 0")            # kinematic (no penetration)
    # No-slip at perturbed interface:
    #   û(1) + ĥ·U₀'(1) = 0  →  Dŵ(1) − ik·U₀'(1)·ĥ = 0
    problem.add_equation("wz(z=1) +1j*k*U0z(z=1)*h = 0")
    # Melting temperature constraint:
    #   θ_l⁰(1) + ĥ·Dθ_l⁰(1) + θ̂_l(1) = 0  →  θ̂_l(1) = ĥ
    problem.add_equation("tl(z=1) - h = 0")
    # Same for the solid side (with d₀ = Λ ⟹ coefficient = 1):
    problem.add_equation("ts(z=1) - h = 0")
    # ═══════════════════════════════════════════════════════
    #  Stefan Condition (9th equation for the scalar ĥ)
    # ═══════════════════════════════════════════════════════
    #
    #  −iσ·Λ·S·ĥ = D_z θ̂_l(1) + (−1/d₀)D_ζ θ̂_s(1)
    #
    #  Since ∂/∂z_s = −(1/d₀)·∂/∂ζ:
    #    ∂θ̂_s/∂z_s|_{interface} = −(1/d₀)·tsz(1)
    #
    #  Rearranged:
    #    (1/d₀)·tsz(1) + tlz(1) + iσ·Λ·S·ĥ = 0
    problem.add_equation(
        "1/d0*tsz(z=1) + tlz(z=1) - 1j*sigma*Lam*S*h = 0"
    )
    # ── solve ─────────────────────────────────────────────
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)
# ═══════════════════════════════════════════════════════════
#  Solve at TWO resolutions for convergence filtering
# ═══════════════════════════════════════════════════════════
N1, N2 = 140,150
logger.info(f"Solving EVP at N = {N1} ...")
ev1 = solve_evp(N1)
logger.info(f"  -> {len(ev1)} raw eigenvalues")
logger.info(f"Solving EVP at N = {N2} ...")
ev2 = solve_evp(N2)
logger.info(f"  -> {len(ev2)} raw eigenvalues")
def convergence_filter(ev_lo, ev_hi, tol=0.01):
    """Keep eigenvalues from ev_lo that have a partner in ev_hi
    within relative tolerance tol."""
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e):
            continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    if len(good) == 0:
        logger.warning("Convergence filter returned nothing -> using all finite")
        return ev_lo[np.isfinite(ev_lo)]
    return ev_lo[np.array(good)]
evals = convergence_filter(ev1, ev2, tol=0.01)
logger.info(f"Retained {len(evals)} / {len(ev1)} converged eigenvalues")
# ═══════════════════════════════════════════════════════════
#  Convert eigenvalue to physical phase speed
#
#  The paper uses diffusive time scale: σ = k·Pe·c̃
#  Physical phase speed: c̃ = σ / (k·Pe)
#  c̃_r ∈ [0, 1] for hydrodynamic modes (bounded by U₀)
# ═══════════════════════════════════════════════════════════
c_tilde = evals / (k * Pe)
cr = c_tilde.real
ci = c_tilde.imag
logger.info(f"c̃ range: Re = [{cr.min():.4f}, {cr.max():.4f}], "
            f"Im = [{ci.min():.4f}, {ci.max():.4f}]")
idx = np.argmax(ci)
logger.info(f"Most unstable mode:  c̃ = {cr[idx]:.6f} {ci[idx]:+.6f}i")
logger.info(f"  (raw sigma = {evals[idx]:.4f})")
# ═══════════════════════════════════════════════════════════
#  Plot eigenvalue spectrum
# ═══════════════════════════════════════════════════════════
Re_eff = Pe / Pr
fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(cr, ci, s=50, marker='o',
           facecolors='none', edgecolors='royalblue',
           linewidths=1.0, zorder=3, label='Eigenmodes')
ax.scatter(cr[idx], ci[idx], s=300, marker='*',
           color='gold', edgecolors='k',
           linewidths=0.8, zorder=5, label='Most Unstable Mode')
ax.annotate(f"({cr[idx]:.4f}, {ci[idx]:.4f})",
            xy=(cr[idx], ci[idx]), xycoords='data',
            xytext=(15, 15), textcoords='offset points',
            fontsize=11, fontweight='bold', color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='gray'))
ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.7)
ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-1.5, 0.2)
ax.set_xlabel(r'Phase Speed  $\tilde{c}_r = \sigma_r / (k \cdot Pe)$', fontsize=14)
ax.set_ylabel(r'Growth Rate  $\tilde{c}_i = \sigma_i / (k \cdot Pe)$', fontsize=14)
ax.set_title(
    f'Poiseuille + Melting Phase-Boundary Spectrum\n'
    f'Ra={Ra:.0f},  Pe={Pe:.0f},  Re={Re_eff:.0f},  '
    f'S={S},  Λ={Lam},  k={k}',
    fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, ls=':', alpha=0.5)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('melting_poiseuille_spectrum.png', dpi=150)
plt.show()
logger.info("Plot saved -> melting_poiseuille_spectrum.png")