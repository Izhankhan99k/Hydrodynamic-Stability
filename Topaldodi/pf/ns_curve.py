import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  EVP Solver Function (Parameterized)
# ═══════════════════════════════════════════════════════════
def solve_evp(N, Re, k, S):
    Ra = 0          # Rayleigh number fixed at 0
    Pr = 1.0        # Prandtl number
    Pe = Re * Pr    # Pe = Re when Pr = 1
    Lam = 0.5       
    m = 0.0         
    gamma2 = k**2 + m**2
    d0 = Lam        
    D2U0 = -8.0     

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

    w, wz, Lw, Lwz = [dist.Field(name=n, bases=basis) for n in ['w', 'wz', 'Lw', 'Lwz']]
    tl, tlz, ts, tsz = [dist.Field(name=n, bases=basis) for n in ['tl', 'tlz', 'ts', 'tsz']]
    h, sigma = dist.Field(name='h'), dist.Field(name='sigma')

    tw1, tw2, tw3, tw4 = [dist.Field(name=f'tw{i}') for i in range(1, 5)]
    ttl1, ttl2 = dist.Field(name='ttl1'), dist.Field(name='ttl2')
    tts1, tts2 = dist.Field(name='tts1'), dist.Field(name='tts2')

    U0 = dist.Field(name='U0', bases=basis)
    U0['g'] = 4.0 * z * (1.0 - z)

    ns = dict(
        Ra=Ra, Pe=Pe, Pr=Pr, S=S, Lam=Lam, k=k, gamma2=gamma2, d0=d0, D2U0=D2U0,
        w=w, wz=wz, Lw=Lw, Lwz=Lwz, tl=tl, tlz=tlz, ts=ts, tsz=tsz, h=h, sigma=sigma, U0=U0, dz=dz, lift=lift,
        tw1=tw1, tw2=tw2, tw3=tw3, tw4=tw4, ttl1=ttl1, ttl2=ttl2, tts1=tts1, tts2=tts2
    )
    variables = [w, wz, Lw, Lwz, tl, tlz, ts, tsz, h, tw1, tw2, tw3, tw4, ttl1, ttl2, tts1, tts2]
    problem = d3.EVP(variables, eigenvalue=sigma, namespace=ns)

    problem.add_equation("wz  - dz(w)                  + lift(tw1)  = 0")
    problem.add_equation("dz(wz) - gamma2*w - Lw       + lift(tw2)  = 0")
    problem.add_equation("Lwz - dz(Lw)                 + lift(tw3)  = 0")
    problem.add_equation("Pr*(dz(Lwz) - gamma2*Lw) - 1j*k*Pe*U0*Lw + 1j*k*Pe*D2U0*w - gamma2*Ra*Pr/Pe*tl + 1j*sigma*Lw + lift(tw4) = 0")
    problem.add_equation("tlz - dz(tl) + lift(ttl1) = 0")
    problem.add_equation("dz(tlz) - gamma2*tl - 1j*k*Pe*U0*tl + Pe*w + 1j*sigma*tl + lift(ttl2) = 0")
    problem.add_equation("tsz - dz(ts) + lift(tts1) = 0")
    problem.add_equation("1/d0**2*dz(tsz) - gamma2*ts + 1j*sigma*ts + lift(tts2) = 0")

    problem.add_equation("w(z=0)  = 0")
    problem.add_equation("wz(z=0) = 0")
    problem.add_equation("tl(z=0) = 0")
    problem.add_equation("ts(z=0) = 0")
    problem.add_equation("w(z=1) + 4j*k*h  = 0")
    problem.add_equation("wz(z=1) = 0")
    problem.add_equation("tl(z=1) - h = 0")
    problem.add_equation("ts(z=1) - h = 0")
    problem.add_equation("-1/d0*tsz(z=1) - tlz(z=1) + 1j*sigma*Lam*S*h = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)

def convergence_filter(ev_lo, ev_hi, Pe, tol=0.05):
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e): continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    if len(good) == 0:
        mask = np.isfinite(ev_lo) & (np.abs(ev_lo) < 10 * Pe)
        return ev_lo[mask]
    return ev_lo[good]

# ═══════════════════════════════════════════════════════════
#  Grid Scan Configurations 
# ═══════════════════════════════════════════════════════════
N1, N2 = 130,140                         # Reduced slightly for faster multi-line looping
Re_vals = np.linspace(1000,30000,5)   # Grid resolution (X-axis)
k_vals = np.linspace(0.5, 4.0, 5)       # Grid resolution (Y-axis)
Re_mesh, k_mesh = np.meshgrid(Re_vals, k_vals)

# Define your sequence of Stefan numbers here
stefan_numbers = [1, 2, 3, 4, 5]

# Generate a continuous color gradient array using a colormap
cmap = plt.cm.plasma
colors = cmap(np.linspace(0.1, 0.85, len(stefan_numbers)))

fig, ax = plt.subplots(figsize=(10, 6.5))

# ═══════════════════════════════════════════════════════════
#  Loop & Plot Line Series
# ═══════════════════════════════════════════════════════════
for s_idx, S_val in enumerate(stefan_numbers):
    logger.info(f"Scanning grid for Stefan Number S = {S_val}...")
    max_ci_grid = np.zeros_like(Re_mesh)

    for i in range(len(k_vals)):
        for j in range(len(Re_vals)):
            Re_curr = Re_mesh[i, j]
            k_curr = k_mesh[i, j]
            
            ev1 = solve_evp(N1, Re_curr, k_curr, S_val)
            ev2 = solve_evp(N2, Re_curr, k_curr, S_val)
            evals = convergence_filter(ev1, ev2, Pe=Re_curr)
            
            if len(evals) > 0:
                ci = evals.imag / (k_curr * Re_curr)
                max_ci_grid[i, j] = np.max(ci)
            else:
                max_ci_grid[i, j] = -np.inf

    # Draw the contour line at c_i = 0 for this specific Stefan number
    contour = ax.contour(Re_mesh, k_mesh, max_ci_grid, levels=[0], 
                         colors=[colors[s_idx]], linewidths=2.5)
    
    # Create matching legend entries
    ax.plot([], [], color=colors[s_idx], lw=2.5, label=f'$S = {S_val}$')

# ═══════════════════════════════════════════════════════════
#  Plot Details
# ═══════════════════════════════════════════════════════════
ax.set_xlabel(r'Reynolds Number ($Re$)', fontsize=13, labelpad=8)
ax.set_ylabel(r'Wavenumber ($k$)', fontsize=13, labelpad=8)
ax.set_title(r'Neutral Stability Boundaries ($c_i = 0$) for $Ra = 0$', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, ls=':', alpha=0.6)
ax.legend(title="Stefan Number", title_fontsize='11', fontsize=10, loc='upper right')

plt.tight_layout()
plt.savefig('stefan_neutral_stability_sequence.png', dpi=400)
plt.show()
logger.info("Sequence plot saved as 'stefan_neutral_stability_sequence.png'")