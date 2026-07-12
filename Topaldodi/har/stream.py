import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# --- PARAMETERS ---
a = 3           # Matches figure caption
G = 0
Bo = 1000.0   
Re_val = 100000
k_val = 20

def build_and_solve_evp(N_res):
    """Helper function to build and solve the EVP at a given resolution."""
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
    return solver, np.array(solver.eigenvalues), phi, Lphi, z

# --- STEP 1: SOLVE AT TWO RESOLUTIONS ---
N1 = 180
N2 = 200

print(f"Solving lower resolution (N={N1})...")
_, ev1, _, _, _ = build_and_solve_evp(N1)

print(f"Solving higher resolution (N={N2})...")
solver2, ev2, phi2, Lphi2, z2 = build_and_solve_evp(N2)

# --- STEP 2: DUAL-GRID CONVERGENCE FILTERING ---
tol = 1e-4
converged_indices_in_N2 = []

for idx, e2 in enumerate(ev2):
    if not np.isfinite(e2): 
        continue
    # Find the distance to the closest eigenvalue in the N1 resolution
    min_dist = np.min(np.abs(ev1 - e2))
    denom = max(np.abs(e2), 1.0)
    
    # If the eigenvalue doesn't shift significantly, it's a physical mode
    if (min_dist / denom) < tol:
        converged_indices_in_N2.append(idx)

if len(converged_indices_in_N2) == 0:
    raise ValueError("No converged modes found! Try relaxing the tolerance 'tol' or increasing resolution.")

# --- STEP 3: SELECT THE MOST UNSTABLE PHYSICAL MODE ---
converged_indices_in_N2 = np.array(converged_indices_in_N2)
# Find the maximum growth rate (imaginary part) among ONLY the converged modes
most_unstable_phys_idx = converged_indices_in_N2[np.argmax(ev2[converged_indices_in_N2].imag)]

print(f"Selected physical mode eigenvalue: c = {ev2[most_unstable_phys_idx]}")

# Set the high-resolution solver state to this verified mode
solver2.set_state(most_unstable_phys_idx)

# --- STEP 4: GENERATE AND PLOT 2D FIELDS ---
phi_z = phi2['g']
vorticity_z = Lphi2['g']

# Create a smooth 2D spatial grid for 1 wavelength (kx from 0 to 2*pi)
kx_1d = np.linspace(0, 2 * np.pi, 300)
KX, Z = np.meshgrid(kx_1d, z2.flatten())

# Construct physical fields: Re{ amplitude(z) * exp(i k x) }
psi_2d = np.real(phi_z[:, None] * np.exp(1j * KX))
vorticity_2d = np.real(vorticity_z[:, None] * np.exp(1j * KX))

# Normalize the disturbance vorticity field
vorticity_norm = vorticity_2d / np.max(np.abs(vorticity_2d))

# --- MATPLOTLIB RENDERING ---
fig, ax = plt.subplots(figsize=(7, 5))

# Normalized vorticity background contours (bwr map goes from Blue [-1] to Red [+1])
contourf_plot = ax.contourf(
    KX, Z, vorticity_norm, 
    levels=np.linspace(-1, 1, 100), 
    cmap='bwr', 
    extend='both'
)

# Superimpose Black Streamlines
ax.contour(KX, Z, psi_2d, levels=15, colors='k', linewidths=1.2)

# Graph layout configurations
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1, 0)
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax.set_yticks([-1.0, -0.5, 0.0])

ax.set_xlabel(r'$kx$', fontsize=12)
ax.set_ylabel(r'$z$', fontsize=12, rotation=0, labelpad=15)
ax.set_title(f'$Re = {int(Re_val)}$; $k = {k_val}$', fontsize=12)
ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# Colorbar matching the right panel layout
cbar = fig.colorbar(contourf_plot, ax=ax, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.show()