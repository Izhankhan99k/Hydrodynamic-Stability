"""
Dedalus v3 — Melting Phase-Boundary Stability: Poiseuille Flow
Sweeps over multiple Peclet numbers and plots spectra on a single graph.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
from matplotlib.cm import get_cmap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  Fixed Physical parameters
# ═══════════════════════════════════════════════════════════
Ra   = 0      # Rayleigh number 
Pr   = 1.0          # Prandtl number
S    = 1000000000      # Stefan number 
Lam  = 0.5          # Λ = (Tm−Tc)/ΔT 
k    = 2.042122         # streamwise wavenumber
m    = 0.0          # spanwise wavenumber

gamma2 = k**2 + m**2
d0     = Lam        

# ═══════════════════════════════════════════════════════════
#  EVP solver function (Now takes Pe as an argument)
# ═══════════════════════════════════════════════════════════
def solve_evp(N, Pe):
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

    w   = dist.Field(name='w',   bases=basis)
    wz  = dist.Field(name='wz',  bases=basis)
    Lw  = dist.Field(name='Lw',  bases=basis)
    Lwz = dist.Field(name='Lwz', bases=basis)
    tl  = dist.Field(name='tl',  bases=basis)
    tlz = dist.Field(name='tlz', bases=basis)
    ts  = dist.Field(name='ts',  bases=basis)
    tsz = dist.Field(name='tsz', bases=basis)

    h = dist.Field(name='h')
    c = dist.Field(name='c')                    # Eigenvalue = Phase speed

    tw1, tw2, tw3, tw4 = [dist.Field(name=f'tw{i}') for i in range(1, 5)]
    ttl1, ttl2 = [dist.Field(name=f'ttl{i}') for i in range(1, 3)]
    tts1, tts2 = [dist.Field(name=f'tts{i}') for i in range(1, 3)]

    # Poiseuille base flow
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

    # OS Eq
    problem.add_equation("wz  - dz(w)              + lift(tw1) = 0")
    problem.add_equation("dz(wz) - gamma2*w - Lw   + lift(tw2) = 0")
    problem.add_equation("Lwz - dz(Lw)             + lift(tw3) = 0")
    problem.add_equation(
        "Pr*(dz(Lwz) - gamma2*Lw)"
        " - 1j*k*Pe*(U0*Lw - U0zz*w)"
        " - gamma2*Ra*Pr/Pe*tl"
        " + 1j*k*Pe*c*Lw + lift(tw4) = 0"
    )

    # Liquid Heat
    problem.add_equation("tlz - dz(tl) + lift(ttl1) = 0")
    problem.add_equation(
        "dz(tlz) - gamma2*tl"
        " - 1j*k*Pe*U0*tl"
        " + Pe*w"
        " + 1j*k*Pe*c*tl + lift(ttl2) = 0"
    )

    # Solid Heat
    problem.add_equation("tsz - dz(ts) + lift(tts1) = 0")
    problem.add_equation(
        "1/d0**2*dz(tsz)"
        " - gamma2*ts"
        " + 1j*k*Pe*c*ts + lift(tts2) = 0"
    )

    # BCs
    problem.add_equation("w(z=0)  = 0")        
    problem.add_equation("wz(z=0) = 0")        
    problem.add_equation("tl(z=0) = 0")        
    problem.add_equation("ts(z=0) = 0")        

    problem.add_equation("w(z=1) = 0")
    problem.add_equation("wz(z=1) - 1j*k*U0z(z=1)*h = 0")  
    problem.add_equation("tl(z=1) - h = 0")
    problem.add_equation("ts(z=1) - h = 0")

    # Stefan
    problem.add_equation(
        "1/d0*tsz(z=1) + tlz(z=1) + 1j*k*Pe*Lam*S*c*h = 0"
    )

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)

def get_converged_spectra(Pe, N1=140, N2=150, tol=1e-5):
    """Solves at two resolutions and filters out spurious modes."""
    logger.info(f"--- Sweeping Pe = {Pe} ---")
    ev1 = solve_evp(N1, Pe)
    ev2 = solve_evp(N2, Pe)
    
    good = []
    for i, e in enumerate(ev1):
        if not np.isfinite(e):
            continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev2 - e)) / denom < tol:
            good.append(i)
            
    if len(good) < 5:
        logger.warning(f"Very few converged modes for Pe={Pe}. Loosening tolerance.")
        good = []
        for i, e in enumerate(ev1):
            if not np.isfinite(e):
                continue
            denom = max(abs(e), 1.0)
            if np.min(np.abs(ev2 - e)) / denom < 1e-3:
                good.append(i)
                
    evals = ev1[np.array(good)]
    logger.info(f"Retained {len(evals)} converged modes for Pe = {Pe}")
    return evals

# ═══════════════════════════════════════════════════════════
#  Main Sweep and Plotting
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    Pe_list = [2000, 5000, 8000, 9000, 12000, 15000, 20000]
    
    # Store results
    spectra = {}
    for Pe in Pe_list:
        spectra[Pe] = get_converged_spectra(Pe, N1=140, N2=150)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate distinct colors
    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in range(len(Pe_list))]
    
    for idx, Pe in enumerate(Pe_list):
        evals = spectra[Pe]
        cr = evals.real
        ci = evals.imag
        
        # Plot all converged modes for this Pe
        ax.scatter(cr, ci, s=30, marker='o', 
                   facecolors='none', edgecolors=colors[idx], 
                   linewidths=1.2, zorder=3, alpha=0.8,
                   label=f'Pe = {Pe}')
        
        # Highlight the most unstable mode for this Pe
        max_idx = np.argmax(ci)
        ax.scatter(cr[max_idx], ci[max_idx], s=150, marker='*', 
                   color=colors[idx], edgecolors='k', zorder=5)

    ax.axhline(0, color='gray', ls='--', lw=1.5, alpha=0.7)

    # Adjust zoom to capture the core branches across all Pe
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-1.5, 0.2)
    
    ax.set_xlabel(r'Phase speed  $\tilde{c}_r$', fontsize=16)
    ax.set_ylabel(r'Growth rate  $\tilde{c}_i$', fontsize=16)
    ax.set_title(
        f'Melting Poiseuille Spectrum Sweep\n'
        f'Ra={Ra:.0f},  Pr={Pr},  S={S},  Λ={Lam},  k={k}',
        fontsize=15)
    
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig('melting_poiseuille_sweep.png', dpi=300)
    logger.info("Saved sweep plot to melting_poiseuille_sweep.png")
    plt.show()
