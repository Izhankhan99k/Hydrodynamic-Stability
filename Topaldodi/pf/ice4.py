
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ═══════════════════════════════════════════════════════════
#  Physical parameters
# ═══════════════════════════════════════════════════════════
Ra   = 0     # Rayleigh number    (Eq 2.16a)
Pe   = 15000.0       # Peclet number      (Eq 2.16b)
Pr   = 1.0          # Prandtl number     (Eq 2.16c)
S    = 1       # Stefan number      (Eq 2.17a)
Lam  = 0.5          # Λ = (Tm−Tc)/ΔT    (Eq 2.17b)
k    = 1.0          # streamwise wavenumber
m    = 0.0          # spanwise wavenumber
gamma2 = k**2 + m**2
d0     = Lam         # from initial heat-flux balance (Eq 2.25)
Re_h   = Pe / ( Pr)   # effective Re based on half-channel-width
logger.info(f"Parameters: Ra={Ra}, Pe={Pe}, Pr={Pr}, S={S}, Λ={Lam}")
logger.info(f"  Re_half={Re_h:.0f}, k={k}, d0={d0}")
# ═══════════════════════════════════════════════════════════
#  EVP solver
# ═══════════════════════════════════════════════════════════
def solve_evp(N):
    """
    Build and solve the coupled EVP at Chebyshev resolution N.
    Eigenvalue = c̃ (physical phase speed).
    """
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
    # ── continuous fields ──
    w   = dist.Field(name='w',   bases=basis)   # vertical velocity
    wz  = dist.Field(name='wz',  bases=basis)   # Dw
    Lw  = dist.Field(name='Lw',  bases=basis)   # (D²−γ²)w
    Lwz = dist.Field(name='Lwz', bases=basis)   # D(Lw)
    tl  = dist.Field(name='tl',  bases=basis)   # liquid temp perturbation
    tlz = dist.Field(name='tlz', bases=basis)   # D(θ_l)
    ts  = dist.Field(name='ts',  bases=basis)   # solid temp perturbation
    tsz = dist.Field(name='tsz', bases=basis)   # D_ζ(θ_s)
    # ── scalar fields ──
    h = dist.Field(name='h')                    # interface amplitude
    c = dist.Field(name='c')                    # eigenvalue (phase speed)
    # ── tau fields (8 BCs → 8 taus) ──
    tw1  = dist.Field(name='tw1')
    tw2  = dist.Field(name='tw2')
    tw3  = dist.Field(name='tw3')
    tw4  = dist.Field(name='tw4')
    ttl1 = dist.Field(name='ttl1')
    ttl2 = dist.Field(name='ttl2')
    tts1 = dist.Field(name='tts1')
    tts2 = dist.Field(name='tts2')
    # ── Poiseuille base flow ──
    # U₀(z) = 4z(1−z),  U₀'(z) = 4−8z,  U₀''(z) = −8
    U0 = dist.Field(name='U0', bases=basis)
    U0['g'] = 4.0 * z * (1.0 - z)
    U0z = dist.Field(name='U0z', bases=basis)
    U0z['g'] = 4.0 - 8.0 * z
    U0zz = dist.Field(name='U0zz', bases=basis)
    U0zz['g'] = -8.0 * np.ones_like(z)
    # ── namespace ──
    ns = dict(
        Ra=Ra, Pe=Pe, Pr=Pr, S=S, Lam=Lam,
        k=k, gamma2=gamma2, d0=d0,
        w=w, wz=wz, Lw=Lw, Lwz=Lwz,
        tl=tl, tlz=tlz, ts=ts, tsz=tsz,
        h=h, c=c, U0=U0, U0z=U0z, U0zz=U0zz,
        dz=dz, lift=lift,
        tw1=tw1, tw2=tw2, tw3=tw3, tw4=tw4,
        ttl1=ttl1, ttl2=ttl2, tts1=tts1, tts2=tts2,
    )
    variables = [w, wz, Lw, Lwz, tl, tlz, ts, tsz, h,
                 tw1, tw2, tw3, tw4, ttl1, ttl2, tts1, tts2]
    problem = d3.EVP(variables, eigenvalue=c, namespace=ns)
    # ═══════════════════════════════════════════════════════
    #  SUBSTITUTION:  iσ  →  ikPe·c   everywhere
    # ═══════════════════════════════════════════════════════
    # ── Orr-Sommerfeld (4th order) ────────────────────────
    #
    # From y-vorticity equation (∂z[x-mom] − ∂x[z-mom]):
    #
    #   Pr(D²−γ²)²w − ikPe·U₀(D²−γ²)w + ikPe·U₀''w
    #        − γ²·RaPr/Pe·θ_l  +  ikPe·c·(D²−γ²)w  =  0
    #
    # In first-order form with Lw = (D²−γ²)w:
    problem.add_equation("wz  - dz(w)              + lift(tw1) = 0")
    problem.add_equation("dz(wz) - gamma2*w - Lw   + lift(tw2) = 0")
    problem.add_equation("Lwz - dz(Lw)             + lift(tw3) = 0")
    problem.add_equation(
        "Pr*(dz(Lwz) - gamma2*Lw)"
        " - 1j*k*Pe*(U0*Lw - U0zz*w)"
        " - gamma2*Ra*Pr/Pe*tl"
        " + 1j*k*Pe*c*Lw"
        " + lift(tw4) = 0"
    )
    # ── Liquid heat (2nd order) ───────────────────────────
    #
    # From Eq (2.12) linearised with Dθ_l⁰ = −1:
    #
    #   (D²−γ²)θ_l − ikPe·U₀·θ_l + Pe·w + ikPe·c·θ_l = 0
    problem.add_equation("tlz - dz(tl) + lift(ttl1) = 0")
    problem.add_equation(
        "dz(tlz) - gamma2*tl"
        " - 1j*k*Pe*U0*tl"
        " + Pe*w"
        " + 1j*k*Pe*c*tl"
        " + lift(ttl2) = 0"
    )
    # ── Solid heat (2nd order, mapped domain) ─────────────
    #
    # zₛ = 1+d₀(1−ζ) ⟹ ∂/∂zₛ = −(1/d₀)∂/∂ζ
    #
    #   (1/d₀²)D_ζ²θ_s − γ²θ_s + ikPe·c·θ_s = 0
    problem.add_equation("tsz - dz(ts) + lift(tts1) = 0")
    problem.add_equation(
        "1/d0**2*dz(tsz)"
        " - gamma2*ts"
        " + 1j*k*Pe*c*ts"
        " + lift(tts2) = 0"
    )
    # ═══════════════════════════════════════════════════════
    #  Boundary Conditions  (8 BCs → 8 taus)
    # ═══════════════════════════════════════════════════════
    # ── z = 0: bottom wall (liquid) + cold wall (solid ζ=0) ──
    problem.add_equation("w(z=0)  = 0")        # no penetration
    problem.add_equation("wz(z=0) = 0")        # no slip (û = (i/k)Dŵ = 0)
    problem.add_equation("tl(z=0) = 0")        # fixed T_h perturbation
    problem.add_equation("ts(z=0) = 0")        # fixed T_c perturbation
    # ── z = 1: melting interface ──
    # Kinematic: w(1+h') ≈ w'(1) = 0
    problem.add_equation("w(z=1) = 0")
    # No-slip at perturbed interface:
    #   U₀(1)+h'U₀'(1)+û(1)=0, U₀(1)=0 ⟹ û(1)=−hU₀'(1)
    #   (i/k)Dŵ(1) = −hU₀'(1) ⟹ Dŵ(1) = −ikU₀'(1)h
    #   ⟹ wz(1) − ik·U₀'(1)·h = 0
    problem.add_equation("wz(z=1) - 1j*k*U0z(z=1)*h = 0")
    # Melting temperature (liquid): θ_l⁰(1)+h·(−1)+θ̂_l(1)=0 ⟹ θ̂_l(1)=h
    problem.add_equation("tl(z=1) - h = 0")
    # Melting temperature (solid): θ̂_s(ζ=1) = (Λ/d₀)h = h  (since d₀=Λ)
    problem.add_equation("ts(z=1) - h = 0")
    # ═══════════════════════════════════════════════════════
    #  Stefan condition  (9th equation for scalar ĥ)
    # ═══════════════════════════════════════════════════════
    #
    # −iσĥ = (1/ΛS)[Dθ̂_l(1) − (∂θ̂_s/∂zₛ)(1)]
    #       = (1/ΛS)[tlz(1) + (1/d₀)tsz(1)]
    #
    # ⟹ (1/d₀)tsz(1) + tlz(1) + iσΛSĥ = 0
    # ⟹ (1/d₀)tsz(1) + tlz(1) + ikPe·ΛS·c·ĥ = 0
    problem.add_equation(
        "1/d0*tsz(z=1) + tlz(z=1) + 1j*k*Pe*Lam*S*c*h = 0"
    )
    # ── solve ──
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)
# ═══════════════════════════════════════════════════════════
#  Solve at TWO resolutions for convergence filtering
# ═══════════════════════════════════════════════════════════
N1, N2 = 96, 128
logger.info(f"Solving EVP at N = {N1} ...")
ev1 = solve_evp(N1)
logger.info(f"  {len(ev1)} raw eigenvalues")
# Diagnostic: print raw eigenvalue range
fin1 = ev1[np.isfinite(ev1)]
logger.info(f"  finite: {len(fin1)},  "
            f"Re=[{fin1.real.min():.4f}, {fin1.real.max():.4f}], "
            f"Im=[{fin1.imag.min():.4f}, {fin1.imag.max():.4f}]")
logger.info(f"Solving EVP at N = {N2} ...")
ev2 = solve_evp(N2)
logger.info(f"  {len(ev2)} raw eigenvalues")
# ═══════════════════════════════════════════════════════════
#  Convergence filter
# ═══════════════════════════════════════════════════════════
def convergence_filter(ev_lo, ev_hi, tol=1e-6):
    """Keep eigenvalues from ev_lo that have a partner in ev_hi."""
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e):
            continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    return good
# Try tight tolerance first, fall back to looser if needed
good = convergence_filter(ev1, ev2, tol=1e-6)
if len(good) < 5:
    logger.warning(f"Only {len(good)} modes at tol=1e-6, trying tol=1e-4")
    good = convergence_filter(ev1, ev2, tol=1e-4)
if len(good) < 5:
    logger.warning(f"Only {len(good)} modes at tol=1e-4, trying tol=1e-2")
    good = convergence_filter(ev1, ev2, tol=1e-2)
if len(good) == 0:
    logger.error("No converged modes found! Plotting all finite modes.")
    good = [i for i, e in enumerate(ev1) if np.isfinite(e)]
evals = ev1[np.array(good)]
logger.info(f"Retained {len(evals)} / {len(ev1)} converged eigenvalues")
cr = evals.real
ci = evals.imag
logger.info(f"  c range: Re=[{cr.min():.6f}, {cr.max():.6f}], "
            f"Im=[{ci.min():.6f}, {ci.max():.6f}]")
idx = np.argmax(ci)
logger.info(f"Most unstable mode: c = {cr[idx]:.6f} + {ci[idx]:.6f}i")
# ═══════════════════════════════════════════════════════════
#  Plot
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(cr, ci, s=45, marker='o',
           facecolors='none', edgecolors='royalblue',
           linewidths=1.0, zorder=3, label='Eigenmodes')
ax.scatter(cr[idx], ci[idx], s=300, marker='*',
           color='gold', edgecolors='k',
           linewidths=0.8, zorder=5, label='Most unstable')
ax.annotate(f"c = ({cr[idx]:.4f}, {ci[idx]:.4f})",
            xy=(cr[idx], ci[idx]), xycoords='data',
            xytext=(15, 15), textcoords='offset points',
            fontsize=10, fontweight='bold', color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color='gray'))
ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.7)
ax.set_xlim(-0.2, 1.5)
ax.set_ylim(-1.5, 0.2)
ax.set_xlabel(r'Phase speed  $\tilde{c}_r$', fontsize=15)
ax.set_ylabel(r'Growth rate  $\tilde{c}_i$', fontsize=15)
ax.set_title(
    f'Melting Poiseuille Spectrum\n'
    f'Ra={Ra:.0f},  Pe={Pe:.0f},  Re_h={Re_h:.0f},  '
    f'S={S},  Λ={Lam},  k={k}',
    fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, ls=':', alpha=0.4)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('melting_poiseuille_spectrum.png', dpi=150)
plt.show()
logger.info("Saved: melting_poiseuille_spectrum.png")
