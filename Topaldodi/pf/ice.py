import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  Fixed Physical Parameters (Half-Width Scaling)
# ═══════════════════════════════════════════════════════════
Ra   = 0                
Pr   = 1.0              
S    = 0.01   
Lam  = 0.5              
k    = 1.02056          # Classical critical half-width wavenumber
m    = 0.0              

gamma2 = k**2 + m**2
d0     = Lam        

# ═══════════════════════════════════════════════════════════
#  EVP Solver Function
# ═══════════════════════════════════════════════════════════
def solve_evp(N, Pe):
    coord = d3.Coordinate('z')
    dist  = d3.Distributor(coord, dtype=np.complex128)
    
    # Keep the physical domain (0, 1) for the liquid phase
    basis = d3.Chebyshev(coord, size=N, bounds=(0, 1))
    z     = dist.local_grid(basis)

    # ─── The Proper Method: Scale the Derivative ───────────
    # Map the [0,1] domain back to the classical half-width scaling
    dz = lambda A: d3.Differentiate(A, coord)
    dy = lambda A: 0.5 * dz(A)  
    # ───────────────────────────────────────────────────────

    try:
        lift_basis = basis.derivative_basis(1)
    except AttributeError:
        lift_basis = basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    # Use 'y' suffix to denote the mapped half-width derivatives
    w   = dist.Field(name='w',   bases=basis)
    wy  = dist.Field(name='wy',  bases=basis)
    Lw  = dist.Field(name='Lw',  bases=basis)
    Lwy = dist.Field(name='Lwy', bases=basis)
    tl  = dist.Field(name='tl',  bases=basis)
    tly = dist.Field(name='tly', bases=basis)
    ts  = dist.Field(name='ts',  bases=basis)
    tsy = dist.Field(name='tsy', bases=basis)

    h = dist.Field(name='h')
    c = dist.Field(name='c')

    tw1, tw2, tw3, tw4 = [dist.Field(name=f'tw{i}') for i in range(1, 5)]
    ttl1, ttl2 = [dist.Field(name=f'ttl{i}') for i in range(1, 3)]
    tts1, tts2 = [dist.Field(name=f'tts{i}') for i in range(1, 3)]

    # Poiseuille base flow mapped to half-width derivatives
    U0 = dist.Field(name='U0', bases=basis)
    U0['g'] = 4.0 * z * (1.0 - z)          # Matches 1 - y^2
    
    U0y = dist.Field(name='U0y', bases=basis)
    U0y['g'] = 2.0 - 4.0 * z               # Matches -2y
    
    U0yy = dist.Field(name='U0yy', bases=basis)
    U0yy['g'] = -2.0 * np.ones_like(z)     # Matches -2

    ns = dict(
        Ra=Ra, Pe=Pe, Pr=Pr, S=S, Lam=Lam,
        k=k, gamma2=gamma2, d0=d0,
        w=w, wy=wy, Lw=Lw, Lwy=Lwy,
        tl=tl, tly=tly, ts=ts, tsy=tsy,
        h=h, c=c, U0=U0, U0y=U0y, U0yy=U0yy,
        dy=dy, lift=lift,
        tw1=tw1, tw2=tw2, tw3=tw3, tw4=tw4,
        ttl1=ttl1, ttl2=ttl2, tts1=tts1, tts2=tts2,
    )

    variables = [w, wy, Lw, Lwy, tl, tly, ts, tsy, h,
                 tw1, tw2, tw3, tw4, ttl1, ttl2, tts1, tts2]

    problem = d3.EVP(variables, eigenvalue=c, namespace=ns)

    # OS Equations (Using mapped half-width derivative 'dy')
    problem.add_equation("wy  - dy(w)              + lift(tw1) = 0")
    problem.add_equation("dy(wy) - gamma2*w - Lw   + lift(tw2) = 0")
    problem.add_equation("Lwy - dy(Lw)             + lift(tw3) = 0")
    problem.add_equation(
        "Pr*(dy(Lwy) - gamma2*Lw)"
        " - 1j*k*Pe*(U0*Lw - U0yy*w)"
        " - gamma2*Ra*Pr/Pe*tl"
        " + 1j*k*Pe*c*Lw + lift(tw4) = 0"
    )

    # Liquid Heat Equation
    problem.add_equation("tly - dy(tl) + lift(ttl1) = 0")
    problem.add_equation(
        "dy(tly) - gamma2*tl"
        " - 1j*k*Pe*U0*tl"
        " + Pe*w"
        " + 1j*k*Pe*c*tl + lift(ttl2) = 0"
    )

    # Solid Heat Equation
    problem.add_equation("tsy - dy(ts) + lift(tts1) = 0")
    problem.add_equation(
        "1/d0**2*dy(tsy)"
        " - gamma2*ts"
        " + 1j*k*Pe*c*ts + lift(tts2) = 0"
    )

    # Boundary Conditions
    problem.add_equation("w(z=0)  = 0")        
    problem.add_equation("wy(z=0) = 0")        
    problem.add_equation("tl(z=0) = 0")        
    problem.add_equation("ts(z=0) = 0")        

    problem.add_equation("w(z=1) = 0")
    problem.add_equation("wy(z=1) - 1j*k*U0y(z=1)*h = 0")  
    problem.add_equation("tl(z=1) - h = 0")
    problem.add_equation("ts(z=1) - h = 0")

    # Stefan Condition
    problem.add_equation(
        "1/d0*tsy(z=1) + tly(z=1) + 1j*k*Pe*Lam*S*c*h = 0"
    )

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)

def get_converged_spectra(Pe, N1=140, N2=150, tol=1e-5):
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

def plot_eigenvalues(spectra_dict, title_params=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('tab10')
    
    for i, (Pe, evals) in enumerate(spectra_dict.items()):
        cr = evals.real
        ci = evals.imag
        color = cmap(i % 10)
        
        ax.scatter(cr, ci, s=40, facecolors='none', edgecolors=color, 
                   linewidths=1.5, alpha=0.8, label=f'Pe = {Pe}')
        
        if len(ci) > 0:
            max_idx = np.argmax(ci)
            ax.scatter(cr[max_idx], ci[max_idx], s=200, marker='*', 
                       color=color, edgecolors='black', zorder=5)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.set_xlabel(r'Phase Speed ($c_r$)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Growth Rate ($c_i$)', fontsize=14, fontweight='bold')
    ax.set_title(f'Eigenvalue Spectrum ($c_i$ vs $c_r$)\n{title_params}', fontsize=15)
    
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-1.5, 0.2)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=11)
    
    plt.tight_layout()
    plt.savefig('eigenvalue_spectrum.png', dpi=300)
    logger.info("Saved plot to eigenvalue_spectrum.png")
    plt.show()

if __name__ == "__main__":
    # Sweeping right around the classical critical value of 5772
    Pe_list = [8000]
    
    spectra = {}
    for Pe in Pe_list:
        spectra[Pe] = get_converged_spectra(Pe, N1=140, N2=150)

    plot_eigenvalues(spectra, title_params=f"Ra={Ra}, Pr={Pr}, S={S:.1e}, Λ={Lam}, k={k}")