import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
from mpi4py import MPI  
# Suppress Dedalus logging output to keep the console clean
logging.getLogger('dedalus').setLevel(logging.WARNING)

def solve_os_evp(alpha_val, Re_val, Nz):
    """Solves the EVP and returns all finite eigenvalues for a given resolution."""
    coord = d3.Coordinate('z')
    dist = d3.Distributor(coord, dtype=np.complex128)
    zbasis = d3.ChebyshevT(coord, size=Nz, bounds=(-1,1))
    z = dist.local_grid(zbasis)

    c = dist.Field(name='c')
    w = dist.Field(name='w', bases=zbasis)
    wz = dist.Field(name='wz', bases=zbasis)
    wzz = dist.Field(name='wzz', bases=zbasis)
    wzzz = dist.Field(name='wzzz', bases=zbasis)
    
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    tau3 = dist.Field(name='tau3')
    tau4 = dist.Field(name='tau4')

    dz = lambda A: d3.Differentiate(A, coord)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    U = dist.Field(name='U', bases=zbasis)
    Uzz = dist.Field(name='Uzz', bases=zbasis)
    U['g'] = 1 - z**2
    Uzz['g'] = -2.0

    alpha = float(alpha_val)
    Re = float(Re_val)

    problem = d3.EVP([w, wz, wzz, wzzz, tau1, tau2, tau3, tau4], eigenvalue=c, namespace=locals())
    problem.add_equation("dz(wzzz) - 2*alpha**2*wzz + alpha**4*w - 1j*alpha*Re*((U-c)*(wzz-alpha**2*w) - Uzz*w) + lift(tau1) = 0")
    problem.add_equation("dz(w) - wz + lift(tau2) = 0")
    problem.add_equation("dz(wz) - wzz + lift(tau3) = 0")
    problem.add_equation("dz(wzz) - wzzz + lift(tau4) = 0")

    problem.add_equation("w(z=-1) = 0")
    problem.add_equation("w(z=+1) = 0")
    problem.add_equation("wz(z=-1) = 0")
    problem.add_equation("wz(z=+1) = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    
    evals = solver.eigenvalues
    return evals[np.isfinite(evals)]

def get_legit_max_growth_rate(alpha, Re, N1=100, N2=125, tolerance=1e-4):
    """
    Solves the EVP at two different resolutions to mathematically eliminate 
    spurious eigenvalues based on grid convergence.
    """
    evals_1 = solve_os_evp(alpha, Re, N1)
    evals_2 = solve_os_evp(alpha, Re, N2)
    
    # Create a 2D distance matrix between the two sets of eigenvalues
    distances = np.abs(evals_1[:, np.newaxis] - evals_2)
    
    # Find the minimum distance for each eigenvalue in set 1
    min_distances = np.min(distances, axis=1)
    
    # A legit eigenvalue will have a distance near zero (it converged)
    legit_mask = min_distances < tolerance
    legit_evals = evals_1[legit_mask]
    
    if len(legit_evals) > 0:
        return np.max(legit_evals.imag)
    else:
        # If no converging modes are found, safely assume the physical system is highly stable
        return -1.0

# --------------------------------------------------
# Parameter Sweep Setup
# --------------------------------------------------

# Coarser grid (20x20) because solving 800 EVPs takes time. 
"""alpha_array = np.linspace(0.8, 1.15, 20)
Re_array = np.linspace(4000, 10000, 20)

A_grid, R_grid = np.meshgrid(alpha_array, Re_array)
max_ci_grid = np.zeros_like(A_grid)

print("Starting rigorous parameter sweep. Solving 800 EVPs. This will take a few minutes...")

total_points = len(Re_array) * len(alpha_array)
count = 0

for i in range(len(Re_array)):
    for j in range(len(alpha_array)):
        # Calculate the rigorous growth rate for this grid point
        max_ci_grid[i, j] = get_legit_max_growth_rate(A_grid[i, j], R_grid[i, j])
        
        count += 1
        if count % 20 == 0:
            print(f"Progress: {count}/{total_points} grid points calculated.")

print("Sweep complete. Generating plot...")

# --------------------------------------------------
# Plotting the Neutral Curve
# --------------------------------------------------"""

"""plt.figure(figsize=(7, 5))

# Plot the neutral curve (where max c_i = 0)
contour = plt.contour(R_grid, A_grid, max_ci_grid, levels=[0.0], colors='black', linewidths=2)

# Shade the unstable region (inside the curve)
plt.contourf(R_grid, A_grid, max_ci_grid, levels=[0.0, np.max(max_ci_grid) + 0.1], colors=['#ff9999'], alpha=0.5)

# Plot the theoretical critical point
plt.plot(5772.22, 1.02056, 'b*', markersize=12, label='Critical Point (Re=5772, α=1.02)')

plt.title("Rigorous Neutral Stability Curve for Plane Poiseuille Flow")
plt.xlabel("Reynolds Number (Re)")
plt.ylabel("Wavenumber (α)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()"""




"""for i in [.5,1,2]:
    for j in [4000,5000,6000]:
        alpha_c=i
        Re_c=j
        print(f"Solving spectrum for critical point Re={Re_c}, alpha={alpha_c}...")
        evals_100= solve_os_evp(alpha_c, Re_c, Nz=100)
        evals_120 = solve_os_evp(alpha_c, Re_c, Nz=120)

        plt.figure(figsize=(8, 8))

        # Plot the two resolutions with different colors and sizes
        plt.scatter(evals_100.real, evals_100.imag, s=40, color='blue', alpha=0.6, label='Nz =100' )
        plt.scatter(evals_120.real, evals_120.imag, s=15, color='red', alpha=0.9, label='Nz = 120')

        plt.axhline(0, color='k', alpha=0.3)
        plt.axvline(0, color='k', alpha=0.3)
        plt.axvline(1, color='k', alpha=0.3)

        plt.xlabel(r"$c_r$ (Phase Speed)")
        plt.ylabel(r"$c_i$ (Growth Rate)")
        plt.title(f"Alpha={alpha_c} and Reynlods={Re_c}")

        # Zoom in on the physical fluid region
        plt.xlim(-1, 2)
        plt.ylim(-2, 0.5)

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()"""


Re_fixed = 8000.0

# We will sweep alpha from 0.6 to 1.2 to capture the whole unstable hump
alpha_array = np.linspace(0.6, 1.2, 40)
growth_rates = np.zeros_like(alpha_array)

print(f"Calculating growth rates for Re={Re_fixed} across {len(alpha_array)} alpha values...")

for i, alpha in enumerate(alpha_array):
    growth_rates[i] = get_legit_max_growth_rate(alpha, Re_fixed)
    if (i + 1) % 10 == 0:
        print(f"Progress: {i + 1}/{len(alpha_array)} points calculated.")

# --------------------------------------------------
# 2. Plotting the Results
# --------------------------------------------------
plt.figure(figsize=(8, 5))

# Plot the curve
plt.plot(alpha_array, growth_rates, 'b-o', linewidth=2, markersize=5, label=f'Re = {Re_fixed}')

# Add a prominent horizontal line at zero to show the stability boundary
plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Neutral Stability ($c_i = 0$)')



plt.title(f"Growth Rate vs. Wavenumber for Plane Poiseuille Flow ($Re={Re_fixed}$)", fontsize=14)
plt.xlabel(r"Wavenumber ($\alpha$)", fontsize=12)
plt.ylabel(r"Max Growth Rate ($c_i$)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()