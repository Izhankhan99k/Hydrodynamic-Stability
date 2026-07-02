import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Parameters (can be modified by user)
Re = 1000.0   # Reynolds number
k = 10.0       # Wavenumber
a = 0    # Curvature parameter
G=-3
Bo = 1000.0   # Bond number
def solve_evp(N_res):
    # Coordinate and basis
    coord = d3.Coordinate('z')
    dist = d3.Distributor(coord, dtype=np.complex128)
    # The domain is from rigid wall (z = -1) to free surface (z = 0)
    basis = d3.Chebyshev(coord, size=N_res, bounds=(-1, 0))
    z = dist.local_grid(basis)
    # Base state velocity and its derivatives
    U = dist.Field(name='U', bases=basis)
    U['g'] = a * z**2 + (a + 1) * z + 1
    Uz = dist.Field(name='Uz', bases=basis)
    Uz['g'] = 2 * a * z + (a + 1)
    Uzz = dist.Field(name='Uzz', bases=basis)
    Uzz['g'] = 2 * a * np.ones_like(z)
    # Fields for the streamfunction perturbation
    phi = dist.Field(name='phi', bases=basis)
    phiz = dist.Field(name='phiz', bases=basis)
    Lphi = dist.Field(name='Lphi', bases=basis)
    Lphiz = dist.Field(name='Lphiz', bases=basis)
    # Free surface displacement amplitude (scalar)
    eta = dist.Field(name='eta')
    # Eigenvalue (complex phase speed)
    c = dist.Field(name='c')
    # Tau fields for tau-method boundary conditions
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
    # Namespace for the equation parser
    ns = dict(
        Re=Re, k=k, a=a, G=G, Bo=Bo,
        phi=phi, phiz=phiz, Lphi=Lphi, Lphiz=Lphiz, eta=eta, c=c,
        tau1=tau1, tau2=tau2, tau3=tau3, tau4=tau4,
        dz=dz, lift=lift, U=U, Uz=Uz, Uzz=Uzz
    )
    # Initialize EVP
    problem = d3.EVP([phi, phiz, Lphi, Lphiz, eta, tau1, tau2, tau3, tau4], eigenvalue=c, namespace=ns)
    # 1. ODEs (first-order reduction for Orr-Sommerfeld)
    problem.add_equation("phiz - dz(phi) + lift(tau1) = 0")
    problem.add_equation("Lphi - dz(phiz) + (k**2)*phi + lift(tau2) = 0")
    problem.add_equation("Lphiz - dz(Lphi) + lift(tau3) = 0")
    
    # Equation 2.11: (U - c)(D^2 - k^2)phi - U''phi = 1/(ikRe) * (D^2 - k^2)^2 phi
    problem.add_equation("1j*k*Re*(U*Lphi - Uzz*phi) - (dz(Lphiz) - (k**2)*Lphi) - 1j*k*Re*c*Lphi + lift(tau4) = 0")
    # 2. Boundary conditions at rigid wall (z = -1)
    problem.add_equation("phi(z=-1) = 0")
    problem.add_equation("phiz(z=-1) = 0")
    # 3. Boundary conditions at free surface (z = 0)
    # (2.13a) Kinematic condition: eta = phi / (c - U)
    problem.add_equation("-eta*U(z=0) - phi(z=0) + c*eta = 0")
    # (2.13b) Tangential stress condition: eta*U'' + phi'' + k^2*phi = 0
    # Note: phi'' = Lphi + k^2*phi -> phi'' + k^2*phi = Lphi + 2*k^2*phi
    problem.add_equation("eta*Uzz(z=0) + Lphi(z=0) + 2*(k**2)*phi(z=0) = 0")
    # (2.14) Normal stress condition: 
    # phi''' - 3k^2 phi' + ikRe(U' phi + (c-U) phi') - ikRe G eta (1 + k^2/Bo) = 0
    # Note: phi''' = Lphiz + k^2*phiz -> phi''' - 3k^2*phiz = Lphiz - 2*k^2*phiz
    problem.add_equation("Lphiz(z=0) - 2*(k**2)*phiz(z=0) + 1j*k*Re*(Uz(z=0)*phi(z=0) - U(z=0)*phiz(z=0)) - 1j*k*Re*G*eta*(1 + (k**2)/Bo) + 1j*k*Re*c*phiz(z=0) = 0")
    # Build solver and solve
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    return np.array(solver.eigenvalues)
logger.info("Solving EVP at N = 128 ...")
ev1 = solve_evp(128)
logger.info("Solving EVP at N = 192 for convergence check ...")
ev2 = solve_evp(192)
def convergence_filter(ev_lo, ev_hi, tol=0.01):
    """Keep eigenvalues from ev_lo that have a partner in ev_hi."""
    good = []
    for i, e in enumerate(ev_lo):
        if not np.isfinite(e):
            continue
        denom = max(abs(e), 1.0)
        if np.min(np.abs(ev_hi - e)) / denom < tol:
            good.append(i)
    # Base filter based on physical boundaries
    phys_mask = np.isfinite(ev_lo)
    
    if len(good) == 0:
        logger.warning("Convergence filter returned nothing → falling back to physical bounds")
        return ev_lo[phys_mask]
        
    # Apply physical mask even to converged modes to remove highly damped diffusion modes
    converged = np.array(good)
    final_mask = phys_mask[converged]
    return ev_lo[converged[final_mask]]
evals = convergence_filter(ev1, ev2, tol=0.01)
logger.info(f"Retained {len(evals)} physical modes.")
cr_clean = evals.real
ci_clean = evals.imag
#fig, ax = plt.subplots(figsize=(10, 6))

# Separate unstable modes (ci > 0) from stable/neutrally stable modes
unstable_mask = ci_clean > 0
stable_mask = ~unstable_mask
np.savez("stability_results.npz", 
          cr_clean =cr_clean ,
          ci_clean =ci_clean ,
         unstable_mask=unstable_mask,
         stable_mask =stable_mask,Re=Re,G=G,Bo=Bo  ,k=k)
         
# Plot stable modes (original open blue circles styling)
'''ax.scatter(cr_clean[stable_mask], ci_clean[stable_mask], 
           marker='o', facecolors='none', edgecolors='blue', label='Stable Eigenmodes')

# Plot unstable modes as solid red dots (ci > 0)
if np.any(unstable_mask):
    ax.scatter(cr_clean[unstable_mask], ci_clean[unstable_mask], 
               color='red', marker='o', s=60, edgecolors='darkred', zorder=4, 
               label='Unstable Modes ($c_i > 0$)')

# Highlight the single most unstable mode with a gold star
if len(cr_clean) > 0:
    idx = np.argmax(ci_clean)
    ax.scatter(cr_clean[idx], ci_clean[idx], color='gold', marker='*', s=250, 
               edgecolors='k', zorder=5, label='Most Unstable Mode')
    
    ax.annotate(f"({cr_clean[idx]:.3f}, {ci_clean[idx]:.3f})", 
                 xy=(cr_clean[idx], ci_clean[idx]), xycoords='data',
                 xytext=(15, 10), textcoords='offset points',
                 fontsize=11, fontweight='bold', color='darkred',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='gray'))
    
    logger.info(f"Most unstable mode: c = {cr_clean[idx]:.4f} {ci_clean[idx]:+.4f}i")

# Plot styling adjustments
ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax.set_xlabel(r'Phase Speed ($c_r$)', fontsize=14)
ax.set_ylabel(r'Growth Rate ($c_i$)', fontsize=14)
ax.set_title(f'Couette-Poiseuille Free-Surface Spectrum\n$Re={Re}, k={k}, a={a}, G={G}, Bo={Bo}$', fontsize=14)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right')
plt.xlim(-1, 2.5)
plt.ylim(-2.5, 1)
plt.tight_layout()
plt.show()'''