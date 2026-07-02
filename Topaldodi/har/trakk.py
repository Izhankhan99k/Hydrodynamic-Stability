import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Fixed parameters
Re = 1000.0   # Reynolds number
a = 0         # Fixed curvature parameter
G = 0
Bo = 1000.0   # Bond number

# --- YOUR UPDATED ARRAY (CORRECTED SYNTAX BLUNDERS) ---
# Added commas between 2.5/3, 3/3.5, and split the 10.010.5 mashup cleanly
k_values = np.arange(0.2, 11, 0.1)
def solve_evp(N_res, k_param):
    coord = d3.Coordinate('z')
    dist = d3.Distributor(coord, dtype=np.complex128)
    basis = d3.Chebyshev(coord, size=N_res, bounds=(-1, 0))
    z = dist.local_grid(basis)
    
    U = dist.Field(name='U', bases=basis)
    U['g'] = a * z**2 + (a + 1) * z + 1
    Uz = dist.Field(name='Uz', bases=basis)
    Uz['g'] = 2 * a * z + (a + 1)
    Uzz = dist.Field(name='Uzz', bases=basis)
    Uzz['g'] = 2 * a * np.ones_like(z)
    
    phi = dist.Field(name='phi', bases=basis)
    phiz = dist.Field(name='phiz', bases=basis)
    Lphi = dist.Field(name='Lphi', bases=basis)
    Lphiz = dist.Field(name='Lphiz', bases=basis)
    eta = dist.Field(name='eta')
    c = dist.Field(name='c')
    
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    tau3 = dist.Field(name='tau3')
    tau4 = dist.Field(name='tau4')
    
    dz = lambda A: d3.Differentiate(A, coord)
    try:
        lift_basis = basis.derivative_basis(1)
    except AttributeError:
        lift_basis = basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    
    ns = dict(
        Re=Re, k=k_param, a=a, G=G, Bo=Bo,
        phi=phi, phiz=phiz, Lphi=Lphi, Lphiz=Lphiz, eta=eta, c=c,
        tau1=tau1, tau2=tau2, tau3=tau3, tau4=tau4,
        dz=dz, lift=lift, U=U, Uz=Uz, Uzz=Uzz
    )
    
    problem = d3.EVP([phi, phiz, Lphi, Lphiz, eta, tau1, tau2, tau3, tau4], eigenvalue=c, namespace=ns)
    problem.add_equation("phiz - dz(phi) + lift(tau1) = 0")
    problem.add_equation("Lphi - dz(phiz) + (k**2)*phi + lift(tau2) = 0")
    problem.add_equation("Lphiz - dz(Lphi) + lift(tau3) = 0")
    problem.add_equation("1j*k*Re*(U*Lphi - Uzz*phi) - (dz(Lphiz) - (k**2)*Lphi) - 1j*k*Re*c*Lphi + lift(tau4) = 0")
    
    problem.add_equation("phi(z=-1) = 0")
    problem.add_equation("phiz(z=-1) = 0")
    problem.add_equation("-eta*U(z=0) - phi(z=0) + c*eta = 0")
    problem.add_equation("eta*Uzz(z=0) + Lphi(z=0) + 2*(k**2)*phi(z=0) = 0")
    problem.add_equation("Lphiz(z=0) - 2*(k**2)*phiz(z=0) + 1j*k*Re*(Uz(z=0)*phi(z=0) - U(z=0)*phiz(z=0)) - 1j*k*Re*G*eta*(1 + (k**2)/Bo) + 1j*k*Re*c*phiz(z=0) = 0")
    
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)

def convergence_filter(ev_lo, ev_hi, tol=0.01):
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e): continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    phys_mask = np.isfinite(ev_lo)
    if len(good) == 0: return ev_lo[phys_mask]
    converged = np.array(good)
    return ev_lo[converged[phys_mask[converged]]]

# --- SMART MULTI-ROW GRID ARRANGEMENT ---
# --- SMART MULTI-ROW GRID ARRANGEMENT ---
num_plots = len(k_values)
cols = 7
rows = int(np.ceil(num_plots / cols))

fig, ax = plt.subplots(figsize=(11, 7))

# Create a continuous colormap normalized to your specific wavenumber range
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=min(k_values), vmax=max(k_values))
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
sm.set_array([])

# Render Pink Unstable Zone (above neutral stability line)
c_i_max = 0.5
ax.axhspan(0, c_i_max, facecolor='#FFE4E6', alpha=0.5, zorder=0, label='Unstable Region')

# Structural Guidelines
ax.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=1)
ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, zorder=1)

# Tracks if we have plotted an unstable mode yet to prevent legend duplication
unstable_legend_added = False

print("Computing and superimposing all profiles onto a single canvas...")
for k_current in k_values:
    ev1 = solve_evp(128, k_current)
    ev2 = solve_evp(192, k_current)
    evals = convergence_filter(ev1, ev2, tol=0.01)
    
    cr = evals.real
    ci = evals.imag
    
    # Calculate the continuous trace color for this specific wavenumber loop
    current_color = sm.to_rgba(k_current)
    
    # Separate stable points from unstable points
    unstable_mask = ci > 0
    stable_mask = ~unstable_mask
    
    # Plot stable points using the continuous colormap gradient
    ax.scatter(cr[stable_mask], ci[stable_mask], marker='o', color=current_color, 
               edgecolors='black', linewidths=0.3, s=25, alpha=0.7, zorder=2)
    
    # Highlight unstable points in bright red with a gold star outline
    if np.any(unstable_mask):
        label_text = 'Unstable Modes ($c_i > 0$)' if not unstable_legend_added else ""
        ax.scatter(cr[unstable_mask], ci[unstable_mask], color='red', marker='*', 
                   s=130, edgecolors='gold', linewidths=0.8, zorder=4, label=label_text)
        unstable_legend_added = True

# Formatting the superimposed canvas
ax.set_title(f"Superimposed Couette-Poiseuille Free-Surface Spectrum\nWavenumber Spectrum Evolution ($k = 0.25 \longrightarrow {max(k_values)}$) | Re = {Re}, a = {a}", 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel(r"Phase Speed ($c_r$)", fontsize=12)
ax.set_ylabel(r"Growth Rate ($c_i$)", fontsize=12)

ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-2.0, c_i_max)
ax.grid(True, linestyle=':', alpha=0.4)

# Add a colorbar to decode which eigenvalue branch belongs to which wavenumber
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r"Wavenumber ($k$)", fontsize=11, fontweight='bold')

ax.legend(loc='lower left', framealpha=0.9)
plt.tight_layout()
plt.savefig('trakk.png')
plt.show()