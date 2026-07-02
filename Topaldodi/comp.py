import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  Physical parameters
# ═══════════════════════════════════════════════════════════
Ra   = 0           # Rayleigh number
Pe   = 40000.0     # Peclet number
Pr   = 7.0         # Prandtl number
S    = 1.0         # Stefan number
Lam  = 0.5         # Lambda = (Tm - Tc) / DeltaT
k    = 1.0         # streamwise wavenumber
m    = 0.0         # spanwise wavenumber
gamma2 = k**2 + m**2
d0     = Lam       # base-state solid thickness

# ═══════════════════════════════════════════════════════════
#  EVP solver function: Phase-Change (Melting)
# ═══════════════════════════════════════════════════════════
def solve_evp_phase_change(N):
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
    
    h     = dist.Field(name='h')
    sigma = dist.Field(name='sigma')
    
    tw1  = dist.Field(name='tw1')
    tw2  = dist.Field(name='tw2')
    tw3  = dist.Field(name='tw3')
    tw4  = dist.Field(name='tw4')
    ttl1 = dist.Field(name='ttl1')
    ttl2 = dist.Field(name='ttl2')
    tts1 = dist.Field(name='tts1')
    tts2 = dist.Field(name='tts2')
    
    U0 = dist.Field(name='U0', bases=basis)
    U0['g'] = 1.0 - z
    
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
    
    problem.add_equation("wz  - dz(w)                  + lift(tw1)  = 0")
    problem.add_equation("dz(wz) - gamma2*w - Lw       + lift(tw2)  = 0")
    problem.add_equation("Lwz - dz(Lw)                 + lift(tw3)  = 0")
    problem.add_equation(
        "Pr*(dz(Lwz) - gamma2*Lw) - 1j*k*Pe*U0*Lw - gamma2*Ra*Pr/Pe*tl + 1j*sigma*Lw + lift(tw4) = 0"
    )
    
    problem.add_equation("tlz - dz(tl) + lift(ttl1) = 0")
    problem.add_equation(
        "dz(tlz) - gamma2*tl - 1j*k*Pe*U0*tl + Pe*w + 1j*sigma*tl + lift(ttl2) = 0"
    )
    
    problem.add_equation("tsz - dz(ts) + lift(tts1) = 0")
    problem.add_equation(
        "1/d0**2*dz(tsz) - gamma2*ts + 1j*sigma*ts + lift(tts2) = 0"
    )
    
    problem.add_equation("w(z=0)  = 0")
    problem.add_equation("wz(z=0) = 0")
    problem.add_equation("tl(z=0) = 0")
    problem.add_equation("ts(z=0) = 0")
    
    problem.add_equation("w(z=1)  = 0")
    problem.add_equation("wz(z=1) + 1j*k*h = 0")
    problem.add_equation("tl(z=1) - h = 0")
    problem.add_equation("ts(z=1) - h = 0")
    
    problem.add_equation("-1/d0*tsz(z=1) - tlz(z=1) + 1j*sigma*Lam*S*h = 0")
    
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)

# ═══════════════════════════════════════════════════════════
#  EVP solver function: Standard Couette (Rigid Walls)
# ═══════════════════════════════════════════════════════════
def solve_evp_standard_couette(N):
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
    sigma = dist.Field(name='sigma')
    
    tw1  = dist.Field(name='tw1')
    tw2  = dist.Field(name='tw2')
    tw3  = dist.Field(name='tw3')
    tw4  = dist.Field(name='tw4')
    
    U0 = dist.Field(name='U0', bases=basis)
    U0['g'] = 1.0 - z
    
    ns = dict(
        Pe=Pe, Pr=Pr, k=k, gamma2=gamma2,
        w=w, wz=wz, Lw=Lw, Lwz=Lwz,
        sigma=sigma, U0=U0, dz=dz, lift=lift,
        tw1=tw1, tw2=tw2, tw3=tw3, tw4=tw4,
    )
    variables = [w, wz, Lw, Lwz, tw1, tw2, tw3, tw4]
    problem = d3.EVP(variables, eigenvalue=sigma, namespace=ns)
    
    problem.add_equation("wz  - dz(w)                  + lift(tw1)  = 0")
    problem.add_equation("dz(wz) - gamma2*w - Lw       + lift(tw2)  = 0")
    problem.add_equation("Lwz - dz(Lw)                 + lift(tw3)  = 0")
    
    # Standard OS equation for isothermal flow (Ra=0 means no buoyancy)
    problem.add_equation(
        "Pr*(dz(Lwz) - gamma2*Lw) - 1j*k*Pe*U0*Lw + 1j*sigma*Lw + lift(tw4) = 0"
    )
    
    problem.add_equation("w(z=0)  = 0")
    problem.add_equation("wz(z=0) = 0")
    problem.add_equation("w(z=1)  = 0")
    problem.add_equation("wz(z=1) = 0")   # Rigid upper wall
    
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)

# ═══════════════════════════════════════════════════════════
#  Convergence filter
# ═══════════════════════════════════════════════════════════
def convergence_filter(ev_lo, ev_hi, tol=0.05):
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e):
            continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    if len(good) == 0:
        mask = np.isfinite(ev_lo) & (np.abs(ev_lo) < 10 * Pe)
        return ev_lo[mask]
    return ev_lo[good]

# ═══════════════════════════════════════════════════════════
#  Solve and compare
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    N1, N2 = 100,120
    
    
    logger.info("Solving Phase-Change EVP...")
    ev_pc_1 = solve_evp_phase_change(N1)
    ev_pc_2 = solve_evp_phase_change(N2)
    evals_pc = convergence_filter(ev_pc_1, ev_pc_2, tol=0.05)
    cr_pc = evals_pc.real / (k * Pe)
    ci_pc = evals_pc.imag / (k * Pe)
    
    logger.info("Solving Standard Couette EVP...")
    ev_std_1 = solve_evp_standard_couette(N1)
    ev_std_2 = solve_evp_standard_couette(N2)
    evals_std = convergence_filter(ev_std_1, ev_std_2, tol=0.05)
    cr_std = evals_std.real / (k * Pe)
    ci_std = evals_std.imag / (k * Pe)
    
      # ═══════════════════════════════════════════════════════════
    #  Plot Comparison
    # ═══════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Standard Couette modes (Larger, hollow circles)
    ax.scatter(cr_std, ci_std, s=90, marker='o',
               facecolors='none', edgecolors='blue',
               linewidths=1.5, zorder=2, alpha=0.8, label='Standard Couette (Rigid Walls)')
               
    # Plot Phase-Change modes (Smaller, filled circles)
    ax.scatter(cr_pc, ci_pc, s=25, marker='o',
               color='red', edgecolors='none',
               zorder=3, alpha=0.9, label='Phase-Change Couette')
              
    
    idx_pc = np.argmax(ci_pc)
    logger.info(f"Most unstable mode (Phase-Change): c_r = {cr_pc[idx_pc]:.4f}, c_i = {ci_pc[idx_pc]:.4f}")
    ax.annotate(f"({cr_pc[idx_pc]:.2f}, {ci_pc[idx_pc]:.2f})", 
            xy=(cr_pc[idx_pc], ci_pc[idx_pc]), xycoords='data',
            xytext=(10, 10), textcoords='offset points',
            fontsize=11, fontweight='bold', color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='gray'))
        
    
    idx_std = np.argmax(ci_std)
    logger.info(f"Most unstable mode (Standard Couette): c_r = {cr_std[idx_std]:.4f}, c_i = {ci_std[idx_std]:.4f}")
    ax.annotate(f"({cr_std[idx_std]:.2f}, {ci_std[idx_std]:.2f})", 
            xy=(cr_std[idx_std], ci_std[idx_std]), xycoords='data',
            xytext=(10, 10), textcoords='offset points',
            fontsize=11, fontweight='bold', color='darkred',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='green'))
    
    ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel(r'Phase Speed  ($c_r$)', fontsize=14)
    ax.set_ylabel(r'Growth Rate  ($c_i$)',  fontsize=14)
    ax.set_title(
        f'Comparison: Phase-Change(Most unstable vs Standard Couette \n'
        f'Ra={Ra:.0f},  Pe={Pe:.1f},  k={k}, Pr={Pr:.1f}',
        fontsize=14)
    
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, ls=':', alpha=0.5)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.xlim(-0.25, 1.2)
    plt.ylim(-3, 1)
    
    # Save the figure
    output_img = 'couette_comparison_spectrum.png'
    plt.savefig(output_img, dpi=500)
    logger.info(f"Comparison plot saved → {output_img}")
    plt.show()
