import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

# Suppress Dedalus logging output
logging.getLogger('dedalus').setLevel(logging.WARNING)

def solve_os_and_get_solver(alpha_val, Re_val, Nz):
    """Builds and solves the EVP for a given resolution."""
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

    alpha, Re = float(alpha_val), float(Re_val)

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
# 1. Setup and Two-Resolution Filter
# --------------------------------------------------
# CRITICAL POINT PARAMETERS
alpha = 1.026
Re = 5772.0
N1 = 100
N2 = 120

print(f"Solving at N={N1}...")
solver1, _, _ = solve_os_and_get_solver(alpha, Re, Nz=N1)
print(f"Solving at N={N2}...")
solver2, z, w = solve_os_and_get_solver(alpha, Re, Nz=N2)

# Extract finite eigenvalues
e1 = solver1.eigenvalues[np.isfinite(solver1.eigenvalues)]
e2 = solver2.eigenvalues[np.isfinite(solver2.eigenvalues)]
e2_indices = np.where(np.isfinite(solver2.eigenvalues))[0]

# Distance matrix filter to find legitimate modes
distances = np.abs(e2[:, np.newaxis] - e1)
legit_mask = np.min(distances, axis=1) < 1e-4

legit_e2 = e2[legit_mask]
legit_idx = e2_indices[legit_mask]

# --------------------------------------------------
# 2. Target Specific Modes from the Image
# --------------------------------------------------
# Mode 1: The TS Mode (Top Left point, critical/neutral mode)
idx_top_rel = np.argmax(legit_e2.imag)

# Mode 2: Right Branch (Approaching phase speed 0.95)
target_right = 0.9 - 0.1j
idx_right_rel = np.argmin(np.abs(legit_e2 - target_right))

# Mode 3: Center Stem (Phase speed ~0.3, deeply damped c_i ~ -0.8)
target_center = 0.7 - 0.5j
idx_center_rel = np.argmin(np.abs(legit_e2 - target_center))

# Store them as (Label, Solver Index, Eigenvalue, Color)
selections = [
    ("Critical TS Mode\n(Top Left)", legit_idx[idx_top_rel], legit_e2[idx_top_rel], 'red'),
    ("Center Mode\n(Right Branch)", legit_idx[idx_right_rel], legit_e2[idx_right_rel], 'purple'),
    ("Damped Wall Mode\n(Center Stem)", legit_idx[idx_center_rel], legit_e2[idx_center_rel], 'green')
]

# -------------------------------------------------
# 3. Verification Plot (Spectrum)
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(legit_e2.real, legit_e2.imag, s=20, color='gray', alpha=0.4, label='Filtered Spectrum')

for label, _, e_val, color in selections:
    # Use split label to avoid giant legend box
    short_label = label.split('\n')[0] 
    plt.scatter(e_val.real, e_val.imag, s=100, color=color, edgecolors='black', label=short_label)

plt.axhline(0, color='k', alpha=0.3, linestyle='--')
plt.xlim(0, 1.1)
plt.ylim(-1.5, 0.1)
plt.title(f"Critical Spectrum (Re={Re}, $\\alpha$={alpha})")
plt.xlabel(r"Phase Speed ($c_r$)")
plt.ylabel(r"Growth Rate ($c_i$)")
plt.legend(loc='lower left')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 4. Intuitive Eigenfunction Visualization
# --------------------------------------------------
# Create 3 subplots side-by-side sharing the same Y axis
fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

for ax, (label, original_idx, e_val, color) in zip(axes, selections):
    # Load the specific eigenvector
    solver2.set_state(original_idx, solver2.subsystems[0])
    
    # Extract and normalize
    w_field = w['g']
    w_norm = w_field / np.max(np.abs(w_field))
    # 2. The Instantaneous Snapshot (Real part)
    ax.plot(w_norm.real, z, color=color, linewidth=2.5, label='')
    ax.plot(w_norm.imag, z, color='purple', linewidth=1.5, label='Instantaneous Wave Imaginary')
    ax.plot( np.sqrt(np.power(w_norm.real, 2) + np.power(w_norm.imag, 2)),z, color='black', linewidth=1.5, label='Magnitude')

    # Formatting
    ax.set_title(f"{label}\n$c = {e_val.real:.3f} {e_val.imag:+.4f}j$", fontsize=11)
    ax.set_xlabel(r"Wave Displacement")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1, 1)
    ax.axvline(0, color='k', alpha=0.3)
    ax.axhline(0, color='k', alpha=0.1, linestyle='--')
    ax.grid(True, alpha=0.3)

# Add Y-axis label only to the first (leftmost) plot
axes[0].set_ylabel(r"Channel Height ($z$)")

# Add a single legend to the first plot
axes[0].legend(loc='lower left', fontsize=9)

plt.suptitle(f"Fluid Displacement Profiles inside the Channel (Re={Re}, $\\alpha$={alpha})", fontsize=15, y=0.98)
plt.tight_layout()
plt.show() 


