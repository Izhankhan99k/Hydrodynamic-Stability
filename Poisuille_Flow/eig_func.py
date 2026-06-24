import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

# Suppress Dedalus logging output
logging.getLogger('dedalus').setLevel(logging.WARNING)

def solve_os_2d(alpha_val, Re_val, Nz):
    """
    Solves the standard 2D Orr-Sommerfeld equation.
    """
    coord = d3.Coordinate('z')
    dist = d3.Distributor(coord, dtype=np.complex128)
    zbasis = d3.ChebyshevT(coord, size=Nz, bounds=(-1, 1))
    z = dist.local_grid(zbasis)

    # Base flow (Plane Poiseuille)
    U = dist.Field(name='U', bases=zbasis)
    Uzz = dist.Field(name='Uzz', bases=zbasis)
    U['g'] = 1 - z**2
    Uzz['g'] = -2.0

    # Parameters
    alpha = float(alpha_val)
    Re = float(Re_val)

    # Variables (w = normal velocity)
    c = dist.Field(name='c')
    w = dist.Field(name='w', bases=zbasis)
    wz = dist.Field(name='wz', bases=zbasis)
    wzz = dist.Field(name='wzz', bases=zbasis)
    wzzz = dist.Field(name='wzzz', bases=zbasis)

    # Tau polynomials for boundaries
    tau1 = dist.Field(name='tau1')
    tau2 = dist.Field(name='tau2')
    tau3 = dist.Field(name='tau3')
    tau4 = dist.Field(name='tau4')

    dz = lambda A: d3.Differentiate(A, coord)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    problem = d3.EVP([w, wz, wzz, wzzz, tau1, tau2, tau3, tau4], eigenvalue=c, namespace=locals())

    # Orr-Sommerfeld Equation
    problem.add_equation("dz(wzzz) - 2*alpha**2*wzz + alpha**4*w - 1j*alpha*Re*((U-c)*(wzz-alpha**2*w) - Uzz*w) + lift(tau1) = 0")
    
    # Reductions
    problem.add_equation("dz(w) - wz + lift(tau2) = 0")
    problem.add_equation("dz(wz) - wzz + lift(tau3) = 0")
    problem.add_equation("dz(wzz) - wzzz + lift(tau4) = 0")

    # Rigid wall boundary conditions
    problem.add_equation("w(z=-1) = 0")
    problem.add_equation("w(z=+1) = 0")
    problem.add_equation("wz(z=-1) = 0")
    problem.add_equation("wz(z=+1) = 0")

    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    
    return solver, z, w

# --------------------------------------------------
# 1. Solve and Filter (Two-Resolution Method)
# --------------------------------------------------
Re = 5772.0
alpha = 1.0

print(f"Solving 2D Orr-Sommerfeld at Re={Re}, alpha={alpha}...")
solver1, _, _ = solve_os_2d(alpha, Re, Nz=80)
solver2, z, w_field = solve_os_2d(alpha, Re, Nz=100)

e1 = solver1.eigenvalues[np.isfinite(solver1.eigenvalues)]
e2 = solver2.eigenvalues[np.isfinite(solver2.eigenvalues)]
e2_indices = np.where(np.isfinite(solver2.eigenvalues))[0]

# Rigorous filter to remove spurious artifacts
distances = np.abs(e2[:, np.newaxis] - e1)
legit_mask = np.min(distances, axis=1) < 1e-4
legit_e2 = e2[legit_mask]
legit_idx = e2_indices[legit_mask]

# --------------------------------------------------
# 2. Select the Specific Branches
# --------------------------------------------------
# A-branch (Least damped wall mode / Tip of the Y)
idx_A = np.argmax(legit_e2.imag)

# P-branch (Center mode, phase speed approaches 1)
target_P = 0.9 - 0.1j
idx_P = np.argmin(np.abs(legit_e2 - target_P))

# S-branch (Highly damped wall mode / Left leg of the Y)
target_S = 0.2 - 0.5j
idx_S = np.argmin(np.abs(legit_e2 - target_S))

modes_to_plot = [
    ("(a) A-branch", legit_idx[idx_A], legit_e2[idx_A]),
    ("(c) P-branch", legit_idx[idx_P], legit_e2[idx_P]),
    ("(e) S-branch", legit_idx[idx_S], legit_e2[idx_S])
]

# --------------------------------------------------
# 3. Plotting the Eigenfunctions
# --------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

for i, (title_label, original_idx, e_val) in enumerate(modes_to_plot):
    # Set the solver to the target mode
    solver2.set_state(original_idx, solver2.subsystems[0])
    
    # Extract w and normalize to max amplitude of 1
    w_data = w_field['g']
    w_norm = w_data / np.max(np.abs(w_data))
    
    ax = axes[i]
    
    # Thick line for Magnitude, thin lines for Real/Imaginary
    ax.plot(z, np.abs(w_norm), 'k-', linewidth=3, label='Magnitude $|w|$')
    ax.plot(z, w_norm.real, 'r-', linewidth=1.2, label=r'Real $\operatorname{Re}(w)$')
    ax.plot(z, w_norm.imag, 'b--', linewidth=1.2, label=r'Imag $\operatorname{Im}(w)$')
    
    # Formatting
    ax.set_title(f"{title_label} ($c_r$={e_val.real:.3f}, $c_i$={e_val.imag:.3f})")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='k', alpha=0.2)
    
    # Put a legend only on the first plot
    if i == 0:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)

# Shared X-axis label
axes[2].set_xlabel("Channel Height ($z$)")

plt.suptitle(f"2D Orr-Sommerfeld Normal Velocity ($Re={Re}, \\alpha={alpha}$)", fontsize=14, y=0.96)
plt.tight_layout()
# Adjust spacing so the legend fits cleanly
plt.subplots_adjust(top=0.90, hspace=0.4) 
plt.show()