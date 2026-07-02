import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
# Base Parameters
Re = 10.0
k = 1.0
a = -3.0
Bo = 1000.0
N = 120  # Sufficient resolution for the interfacial mode
G_vals = np.arange(0.1, 5.1, 0.1)
tracked_cr = []
tracked_ci = []
actual_G = []
def solve_for_G(G_val):
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
        Re=Re, k=k, a=a, G=G_val, Bo=Bo,
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
    
    # Filter physical modes (c_i > -50 to avoid highly damped diffusion modes)
    mask = np.isfinite(evals) & (ci > -50.0)
    cr = cr[mask]
    ci = ci[mask]
    
    if len(cr) > 0:
        # The interfacial downstream mode is always the one with the maximum phase speed (c_r)
        # Because the internal modes are bounded by U_max = 1
        idx = np.argmax(cr)
        return cr[idx], ci[idx]
    return None, None
print(f"Tracking the interfacial mode from G={G_vals[0]} to G={G_vals[-1]}...")
for G in G_vals:
    c_r, c_i = solve_for_G(G)
    if c_r is not None:
        tracked_cr.append(c_r)
        tracked_ci.append(c_i)
        actual_G.append(G)
        print(f"G = {G:.1f}  ->  c = {c_r:.4f} + {c_i:.4f}i")
# Plot the trajectory of the mode in the (c_r, c_i) plane
fig, ax = plt.subplots(figsize=(10, 6))
# Plot the line connecting the points
ax.plot(tracked_cr, tracked_ci, color='gray', linestyle='--', zorder=1)
# Scatter plot colored by the value of G
scatter = ax.scatter(tracked_cr, tracked_ci, c=actual_G, cmap='plasma', s=80, edgecolor='k', zorder=2)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Gravity Parameter ($G$)', fontsize=12)
# Mark the start and end points
ax.scatter(tracked_cr[0], tracked_ci[0], color='green', marker='*', s=200, edgecolor='k', zorder=3, label=f'Start (G={actual_G[0]:.1f})')
ax.scatter(tracked_cr[-1], tracked_ci[-1], color='red', marker='X', s=150, edgecolor='k', zorder=3, label=f'End (G={actual_G[-1]:.1f})')
ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel(r'Phase Speed ($c_r$)', fontsize=14)
ax.set_ylabel(r'Growth Rate ($c_i$)', fontsize=14)
ax.set_title(f'Trajectory of Interfacial Mode as G varies\n$Re={Re}, k={k}, a={a}, Bo={Bo}$', fontsize=14)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.xlim(0,3)
plt.ylim(-5,1)
plt.savefig("interfacialmode.png")
print("Saved tracking plot to interfacial_mode_tracking.png")
