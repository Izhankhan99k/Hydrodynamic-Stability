import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
from matplotlib.colors import Normalize
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# Base Parameters
a = 0.0       # Linear velocity profile (Couette)
G = 0.0       # Inverse squared Froude number (G=0 for the classic V-shape mentioned in paper)
Bo = 1000.0   # Bond number
N = 64        # Reduced resolution for faster parameter sweep
# Grid for parameter sweep
# The V-shape typically appears at moderate Re and O(1) wavenumbers
num_Re = 40
num_k = 40
Re_vals = np.logspace(1, 4.5, num_Re)
k_vals = np.logspace(-1, 1, num_k)
# Meshgrid for plotting
K_mesh, Re_mesh = np.meshgrid(k_vals, Re_vals)
max_ci = np.zeros_like(K_mesh)
def compute_max_growth_rate(Re_val, k_val):
    """Solve the EVP for a given (Re, k) and return the maximum physical growth rate c_i."""
    coord = d3.Coordinate('z')
    dist = d3.Distributor(coord, dtype=np.complex128)
    basis = d3.Chebyshev(coord, size=N, bounds=(-1, 0))
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
    tau1, tau2, tau3, tau4 = [dist.Field(name=f'tau{i}') for i in range(1, 5)]
    dz = lambda A: d3.Differentiate(A, coord)
    try:
        lift_basis = basis.derivative_basis(1)
    except AttributeError:
        lift_basis = basis
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    ns = dict(
        Re=Re_val, k=k_val, a=a, G=G, Bo=Bo,
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
    
    evals = np.array(solver.eigenvalues)
    cr = evals.real
    ci = evals.imag
    
    # Filter physical modes
    mask = np.isfinite(evals) & (np.abs(cr) < 10) & (ci > -50.0)
    
    if np.any(mask):
        return np.max(ci[mask])
    else:
        return -1.0
print(f"Sweeping parameter space: {num_Re}x{num_k} grid...")
total = num_Re * num_k
count = 0
for i in range(num_Re):
    for j in range(num_k):
        max_ci[i, j] = compute_max_growth_rate(Re_mesh[i, j], K_mesh[i, j])
        count += 1
        if count % 100 == 0:
            print(f"Progress: {count}/{total}")
# Plotting the growth rate contours and neutral stability curve
fig, ax = plt.subplots(figsize=(9, 7))
# Contour plot of growth rates
contourf = ax.contourf(K_mesh, Re_mesh, max_ci, levels=50, cmap='viridis', norm=Normalize(vmin=-0.05, vmax=max_ci.max()))
plt.colorbar(contourf, ax=ax, label=r'Max Growth Rate ($c_i$)')
# Extract and highlight the neutral stability curve (c_i = 0)
neutral_curve = ax.contour(K_mesh, Re_mesh, max_ci, levels=[0.0], colors='red', linewidths=2.5)
# Label the unstable region
ax.text(0.8, 1000, 'Unstable\nRegion', color='red', fontsize=12, fontweight='bold', ha='center')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Wavenumber ($k$)', fontsize=14)
ax.set_ylabel(r'Reynolds Number ($Re$)', fontsize=14)
ax.set_title(f'Neutral Stability Curve (V-Shape)\nLinear Profile ($a=0$), $G={G}, Bo={Bo}$', fontsize=14)
ax.grid(True, which='both', linestyle=':', alpha=0.5)
# Add custom legend for the neutral line
ax.plot([], [], color='red', linewidth=2.5, label='Neutral Curve ($c_i=0$)')
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig('neutral_stability_curve.png', dpi=150)
print("Saved neutral stability curve to neutral_stability_curve.png")
