import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import dedalus.public as d3
import logging

# Suppress Dedalus logging
logging.getLogger('dedalus').setLevel(logging.WARNING)

def solve_os_2d(alpha_val, Re_val, Nz):
    """Solves the 2D Orr-Sommerfeld equation and returns fields."""
    coord = d3.Coordinate('z')
    dist = d3.Distributor(coord, dtype=np.complex128)
    zbasis = d3.ChebyshevT(coord, size=Nz, bounds=(-1, 1))
    z = dist.local_grid(zbasis)

    U = dist.Field(name='U', bases=zbasis)
    Uzz = dist.Field(name='Uzz', bases=zbasis)
    U['g'] = 1 - z**2
    Uzz['g'] = -2.0

    alpha = float(alpha_val)
    Re = float(Re_val)

    c = dist.Field(name='c')
    w = dist.Field(name='w', bases=zbasis)
    wz = dist.Field(name='wz', bases=zbasis)
    wzz = dist.Field(name='wzz', bases=zbasis)
    wzzz = dist.Field(name='wzzz', bases=zbasis)

    tau1, tau2, tau3, tau4 = (dist.Field(name=f'tau{i}') for i in range(1, 5))

    dz = lambda A: d3.Differentiate(A, coord)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)

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
    
    return solver, z, w, wz, wzz

# --------------------------------------------------
# 1. Solve and Filter
# --------------------------------------------------
Re = 5000.0
alpha = 1.0

print("Solving OS equation...")
solver1, _, _, _, _ = solve_os_2d(alpha, Re, Nz=80)
solver2, z_grid, w_field, wz_field, wzz_field = solve_os_2d(alpha, Re, Nz=100)

e1 = solver1.eigenvalues[np.isfinite(solver1.eigenvalues)]
e2 = solver2.eigenvalues[np.isfinite(solver2.eigenvalues)]
e2_indices = np.where(np.isfinite(solver2.eigenvalues))[0]

distances = np.abs(e2[:, np.newaxis] - e1)
legit_mask = np.min(distances, axis=1) < 1e-4
legit_e2 = e2[legit_mask]
legit_idx = e2_indices[legit_mask]

# --------------------------------------------------
# 2. Extract 2D Spatial Fields for A, P, and S
# --------------------------------------------------
# Dedalus Chebyshev grids go from +1 to -1. We sort them to go -1 to +1 for plotting.
z_vals = z_grid.flatten()
sort_idx = np.argsort(z_vals)
z_sorted = z_vals[sort_idx]

targets = [
    ("A-branch (Wall Mode)", 0.1 - 0.05j),
    ("P-branch (Center Mode)", 0.9 - 0.1j),
    ("S-branch (Damped Mode)", 0.1 - 0.5j)
]

modes_data = []

for title, target in targets:
    # Find the nearest valid eigenvalue
    idx = np.argmin(np.abs(legit_e2 - target))
    c_val = legit_e2[idx]
    
    # Load eigenvector state
    solver2.set_state(legit_idx[idx], solver2.subsystems[0])
    
    w_hat = w_field['g'][sort_idx]
    wz_hat = wz_field['g'][sort_idx]
    wzz_hat = wzz_field['g'][sort_idx]
    
    # Normalize the wave amplitude
    norm = 1.0 / np.max(np.abs(w_hat))
    w_hat *= norm
    wz_hat *= norm
    wzz_hat *= norm
    
    # Calculate Perturbation Vorticity: omega_y = dz(u) - dx(w)
    # Note: u = (i/alpha)*wz, so dz(u) = (i/alpha)*wzz
    # dx(w) = i*alpha*w
    vort_hat = 1j * (wzz_hat / alpha - alpha * w_hat)
    
    modes_data.append({
        'title': title,
        'c': c_val,
        'vort_hat': vort_hat,
        'Vmax': np.max(np.abs(vort_hat))
    })

# --------------------------------------------------
# 3. Setup Animation Grids
# --------------------------------------------------
# X goes from 0 to two full wavelengths
x = np.linspace(0, 2 * (2 * np.pi / alpha), 150)
X_mesh, Z_mesh = np.meshgrid(x, z_sorted)

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

# --------------------------------------------------
# 4. Animation Loop
# --------------------------------------------------
# Note: We animate using ONLY the real part of phase speed (c.real). 
# We intentionally ignore the exponential decay (c.imag) so the wave 
# doesn't vanish while you are trying to watch its shape!
frames = 120
time_vals = np.linspace(0, 15, frames)

def update(frame):
    t = time_vals[frame]
    
    for ax, mode in zip(axes, modes_data):
        # Clear previous contours
        for c in ax.collections:
            c.remove()
            
        # Calculate wave phase at time t
        phase = np.exp(1j * alpha * (X_mesh - mode['c'].real * t))
        
        # Multiply shape by phase and take Real part to get instantaneous physical wave
        vort_2d = np.real(mode['vort_hat'][:, None] * phase)
        
        # Plot Vorticity Contours
        ax.contourf(X_mesh, Z_mesh, vort_2d, levels=30, cmap='RdBu_r', 
                    vmin=-mode['Vmax']*0.8, vmax=mode['Vmax']*0.8)
        
        # Formatting
        c_r, c_i = mode['c'].real, mode['c'].imag
        ax.set_title(f"{mode['title']}  |  Phase Speed: $c_r={c_r:.2f}$", fontsize=11)
        ax.set_ylabel("z")

    axes[2].set_xlabel("Downstream Distance (x)")
    return axes

print("Generating animation... (This will pop up in a new window)")
plt.tight_layout()
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

# To save the animation to an MP4 or GIF, uncomment one of these lines:
ani.save('fluid_modes.gif', writer='pillow', fps=20)
# ani.save('fluid_modes.mp4', writer='ffmpeg', fps=20)

plt.show()