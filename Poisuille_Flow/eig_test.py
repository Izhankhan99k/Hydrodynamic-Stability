import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
from mpi4py import MPI   
# Suppress logging
logging.getLogger('dedalus').setLevel(logging.WARNING)

def solve_os_and_get_solver(alpha_val, Re_val, Nz):
    """
    Builds and solves the EVP. 
    Returns the solver, the grid (z), and the wave field object (w).
    """
    coord = d3.Coordinate('z')
    dist = d3.Distributor(coord, dtype=np.complex128)
    zbasis = d3.ChebyshevT(coord, size=Nz, bounds=(-1,1))
    z = dist.local_grid(zbasis)

    c = dist.Field(name='c')
    w    = dist.Field(name='w',    bases=zbasis)
    wz   = dist.Field(name='wz',   bases=zbasis)
    wzz  = dist.Field(name='wzz',  bases=zbasis)
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
    
    return solver, z, w

# --------------------------------------------------
# 1. Setup Parameters 
# --------------------------------------------------
alpha = 1.0
Re = 10000.0
N1 = 100
N2 = 120
tolerance = 1e-4

print(f"Solving EVP at N={N1}...")
solver1, _, _ = solve_os_and_get_solver(alpha, Re, N1)
evals1 = solver1.eigenvalues

print(f"Solving EVP at N={N2}...")
solver2, z, w = solve_os_and_get_solver(alpha, Re, N2)
evals2 = solver2.eigenvalues

# --------------------------------------------------
# 2. Two-Resolution Filter & Index Mapping
# --------------------------------------------------
# Filter out pure mathematical NaNs/Infs first
valid_mask1 = np.isfinite(evals1)
valid_mask2 = np.isfinite(evals2)

e1 = evals1[valid_mask1]
e2 = evals2[valid_mask2]

# Keep track of where e2 values originally lived in solver2's memory
e2_original_indices = np.where(valid_mask2)[0]

# Calculate distance between every eigenvalue in N2 and every eigenvalue in N1
distances = np.abs(e2[:, np.newaxis] - e1)
min_distances = np.min(distances, axis=1)

# Find modes in N2 that have a twin in N1 (meaning they are physically real)
legit_mask = min_distances < tolerance
legit_e2_values = e2[legit_mask]

# Map the legit modes back to their original index in the N2 solver
legit_original_indices = e2_original_indices[legit_mask]

# Sort the legit indices by growth rate (imaginary part) from most unstable to most stable
sort_order = np.argsort(legit_e2_values.imag)[::-1]
final_sorted_indices = legit_original_indices[sort_order]
final_sorted_evals = legit_e2_values[sort_order]

print(f"Found {len(final_sorted_evals)} physically legitimate modes.")

# --------------------------------------------------
# 3. Select and Plot the Modes
# --------------------------------------------------
# We grab the most unstable mode, and two stable ones sitting deeper in the spectrum
mode_selections = [
    ("Unstable TS Mode", final_sorted_indices[0], final_sorted_evals[0]), 
    ("Stable Mode 1", final_sorted_indices[1], final_sorted_evals[1]),
    ("Stable Mode 2", final_sorted_indices[3], final_sorted_evals[3])
]

plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)

colors = ['red', 'blue', 'green']
styles = ['-', '--', ':']

for i, (label, idx, eigenvalue) in enumerate(mode_selections):
    
    # Instruct solver2 to load the specific eigenvector into 'w'
    solver2.set_state(idx, solver2.subsystems[0])
    
    c_r = eigenvalue.real
    c_i = eigenvalue.imag
    
    # Extract the complex wave field and normalize its maximum amplitude to 1
    w_field = w['g']
    w_norm = w_field / np.max(np.abs(w_field))
    
    plot_label = f"{label}\n$c_r$={c_r:.3f}, $c_i$={c_i:.4f}"
    
    # Plot Magnitude |w|
    ax1.plot(z, np.abs(w_norm), color=colors[i], linestyle=styles[i], linewidth=2, label=plot_label)
    
    # Plot Real Part Re(w)
    ax2.plot(z, w_norm.real, color=colors[i], linestyle=styles[i], linewidth=2)

# Formatting
ax1.set_title(r"Wave Amplitude Profile $|w(z)|$")
ax1.set_ylabel(r"$|w|$ (Normalized)")
ax1.set_xlabel(r"Channel Height ($z$)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))

ax2.set_title(r"Wave Oscillation Profile (Real Part)")
ax2.set_xlabel(r"Channel Height ($z$)")
ax2.axvline(0, color='k', alpha=0.1)
ax2.axhline(0, color='k', alpha=0.3)
ax2.grid(True, alpha=0.3)

plt.suptitle(f"Rigorous Orr-Sommerfeld Eigenfunctions (Re={Re}, α={alpha})", fontsize=14)
plt.tight_layout()
plt.show()