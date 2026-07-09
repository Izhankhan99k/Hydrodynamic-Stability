"""
Dedalus v3 — Melting Phase-Boundary Stability: Poiseuille Flow
══════════════════════════════════════════════════════════════════
Toppaladoddi & Wettlaufer (2019), with Couette → Poiseuille replacement.
Physical setup
──────────────
  Liquid  z ∈ [0, 1]     (hot wall z=0, T_h → θ=1;  interface z=1, T_m → θ=0)
  Solid   z_s ∈ [1, 1+d₀] (interface z_s=1, θ=0;  cold wall z_s=1+d₀, θ=−Λ)
  Solid mapped to ζ ∈ [0,1] via  z_s = 1 + d₀(1−ζ)
      ζ=0 → cold wall,  ζ=1 → interface
Eigenvalue
──────────
  c̃ = σ/(k·Pe)  (physical phase speed, directly O(1))
  iσ = ikPe·c̃  everywhere in the equations to guarantee numerical stability
  Perturbations ~ exp(i(kx + my - σt)),  Im(c̃) > 0 → unstable
"""
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ═══════════════════════════════════════════════════════════
#  Physical parameters
# ═══════════════════════════════════════════════════════════
Ra   = 0     # Rayleigh number (Active thermal coupling)
Pe   = 14000.44      # Peclet number (Re_half = 10000, guaranteed unstable zone)
Pr   = 1.0          # Prandtl number
S    = 100000000        # Stefan number (Active melting interface)
Lam  = 0.5          # Λ = (Tm−Tc)/ΔT 
k    = 2.04122          # streamwise wavenumber (Classic alpha=1.0)
m    = 0.0          # spanwise wavenumber
gamma2 = k**2 + m**2
d0     = Lam        # from initial heat-flux balance

logger.info(f"Parameters: Ra={Ra}, Pe={Pe}, Pr={Pr}, S={S}, Λ={Lam}")

# ═══════════════════════════════════════════════════════════
#  EVP solver
# ═══════════════════════════════════════════════════════════
def solve_evp(N):
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
    w   = dist.Field(name='w',   bases=basis)
    wz  = dist.Field(name='wz',  bases=basis)
    Lw  = dist.Field(name='Lw',  bases=basis)
    Lwz = dist.Field(name='Lwz', bases=basis)
    tl  = dist.Field(name='tl',  bases=basis)
    tlz = dist.Field(name='tlz', bases=basis)
    ts  = dist.Field(name='ts',  bases=basis)
    tsz = dist.Field(name='tsz', bases=basis)
    # ── scalar fields ──
    h = dist.Field(name='h')
    c = dist.Field(name='c')                    # Eigenvalue = Phase speed
    # ── tau fields (8 BCs → 8 taus) ──
    tw1  = dist.Field(name='tw1')
    tw2  = dist.Field(name='tw2')
    tw3  = dist.Field(name='tw3')
    tw4  = dist.Field(name='tw4')
    ttl1 = dist.Field(name='ttl1')
    ttl2 = dist.Field(name='ttl2')
    tts1 = dist.Field(name='tts1')
    tts2 = dist.Field(name='tts2')
    # ── Poiseuille base flow (Dynamic Fields) ──
    U0 = dist.Field(name='U0', bases=basis)
    U0['g'] = 4.0 * z * (1.0 - z)
    U0z = dist.Field(name='U0z', bases=basis)
    U0z['g'] = 4.0 - 8.0 * z
    U0zz = dist.Field(name='U0zz', bases=basis)
    U0zz['g'] = -8.0 * np.ones_like(z)
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
    # ── Orr-Sommerfeld (4th order) ────────────────────────
    problem.add_equation("wz  - dz(w)              + lift(tw1) = 0")
    problem.add_equation("dz(wz) - gamma2*w - Lw   + lift(tw2) = 0")
    problem.add_equation("Lwz - dz(Lw)             + lift(tw3) = 0")
    problem.add_equation(
        "Pr*(dz(Lwz) - gamma2*Lw)"
        " - 1j*k*Pe*(U0*Lw - U0zz*w)"
        " - gamma2*Ra*Pr/Pe*tl"
        " + 1j*k*Pe*c*Lw"               # Note: +iσ Lw becomes +ikPe*c*Lw
        " + lift(tw4) = 0"
    )
    # ── Liquid heat (2nd order) ───────────────────────────
    problem.add_equation("tlz - dz(tl) + lift(ttl1) = 0")
    problem.add_equation(
        "dz(tlz) - gamma2*tl"
        " - 1j*k*Pe*U0*tl"
        " + Pe*w"
        " + 1j*k*Pe*c*tl"
        " + lift(ttl2) = 0"
    )
    # ── Solid heat (2nd order, mapped domain) ─────────────
    problem.add_equation("tsz - dz(ts) + lift(tts1) = 0")
    problem.add_equation(
        "1/d0**2*dz(tsz)"
        " - gamma2*ts"
        " + 1j*k*Pe*c*ts"
        " + lift(tts2) = 0"
    )
    # ═══════════════════════════════════════════════════════
    #  Boundary Conditions
    # ═══════════════════════════════════════════════════════
    problem.add_equation("w(z=0)  = 0")        
    problem.add_equation("wz(z=0) = 0")        
    problem.add_equation("tl(z=0) = 0")        
    problem.add_equation("ts(z=0) = 0")        
    problem.add_equation("w(z=1) = 0")
    problem.add_equation("wz(z=1) - 1j*k*U0z(z=1)*h = 0")   # Dynamic shear
    problem.add_equation("tl(z=1) - h = 0")
    problem.add_equation("ts(z=1) - h = 0")
    # ═══════════════════════════════════════════════════════
    #  Stefan condition
    # ═══════════════════════════════════════════════════════
    problem.add_equation(
        "1/d0*tsz(z=1) + tlz(z=1) + 1j*k*Pe*Lam*S*c*h = 0"  # All terms positive!
    )
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)
# ═══════════════════════════════════════════════════════════
#  Solve at TWO resolutions for convergence filtering
# ═══════════════════════════════════════════════════════════
N1, N2 = 240,250  # High resolution for high Pe stiffness
logger.info(f"Solving EVP at N = {N1} ...")
ev1 = solve_evp(N1)
logger.info(f"Solving EVP at N = {N2} ...")
ev2 = solve_evp(N2)
# ═══════════════════════════════════════════════════════════
#  Convergence filter
# ═══════════════════════════════════════════════════════════
def convergence_filter(ev_lo, ev_hi, tol=1e-6):
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e):
            continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    return good
good = convergence_filter(ev1, ev2, tol=1e-6)
if len(good) < 5:
    good = convergence_filter(ev1, ev2, tol=1e-4)
evals = ev1[np.array(good)]
logger.info(f"Retained {len(evals)} converged eigenvalues")
cr = evals.real
ci = evals.imag
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
    f'Ra={Ra:.0f},    '
    f'S={S},  Λ={Lam},  k={k}',
    fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, ls=':', alpha=0.4)
ax.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('melting_poiseuille_spectrum.png', dpi=150)
plt.show()
logger.info("Saved: melting_poiseuille_spectrum.png")