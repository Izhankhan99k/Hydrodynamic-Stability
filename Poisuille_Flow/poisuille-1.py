import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

# Suppress logging
logging.getLogger('dedalus').setLevel(logging.WARNING)

def plot_c_spectrum():
    # Textbook parameters for the classic "Y-shaped" Poiseuille spectrum
    Re = 5000.0
    k_val = 1.0256
    j = 1j
    N = 128  # High resolution to capture the Y-branches cleanly

    print(f"Solving Orr-Sommerfeld EVP for Re={Re}, k={k_val}, N={N}...")

    # Domain: [-1, 1]
    zcoord = d3.Coordinate('z')
    dist = d3.Distributor(zcoord, dtype=np.complex128)
    basis = d3.ChebyshevT(zcoord, size=N, bounds=(-1, 1))
    z = dist.local_grid(basis)

    # Base State: U = 1 - z^2
    U = dist.Field(name='U', bases=basis)
    U['g'] = 1 - z**2
    Uzz = dist.Field(name='Uzz', bases=basis)
    Uzz['g'] = -2 * np.ones_like(z)

    # Fields & Tau Polynomials
    w = dist.Field(name='w', bases=basis)
    wz = dist.Field(name='wz', bases=basis)
    wzz = dist.Field(name='wzz', bases=basis)
    wzzz = dist.Field(name='wzzz', bases=basis)
    sigma = dist.Field(name='sigma')

    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')
    tau_3 = dist.Field(name='tau_3')
    tau_4 = dist.Field(name='tau_4')

    dz = lambda A: d3.Differentiate(A, zcoord)
    lift = lambda A: d3.Lift(A, basis.derivative_basis(1), -1)

    problem = d3.EVP([w, wz, wzz, wzzz, tau_1, tau_2, tau_3, tau_4], eigenvalue=sigma, namespace=locals())

    problem.add_equation("dz(w) - wz + lift(tau_1) = 0")
    problem.add_equation("dz(wz) - wzz + lift(tau_2) = 0")
    problem.add_equation("dz(wzz) - wzzz + lift(tau_3) = 0")
    problem.add_equation(
        "sigma*(wzz - k_val**2 * w) "
        "+ j*k_val*U*(wzz - k_val**2 * w) - j*k_val*Uzz*w "
        "- (1/Re)*(dz(wzzz) - 2*k_val**2 * wzz + k_val**4 * w) + lift(tau_4) = 0"
    )

    problem.add_equation("w(z=-1) = 0")
    problem.add_equation("wz(z=-1) = 0")
    problem.add_equation("w(z=1) = 0")
    problem.add_equation("wz(z=1) = 0")

    # Solve
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])

    # 1. Extract raw eigenvalues (sigma)
    evals = solver.eigenvalues
    evals = evals[np.isfinite(evals)]

    # 2. Convert to Wave Speed (c)
    # sigma = -i * k * c  ==>  c = i * sigma / k
    c_r = -evals.imag / k_val
    c_i = evals.real / k_val

    # 3. Filter the numerical graveyard 
    # (We only care about the physical branches near the top)
    mask = (c_i > -1.0) & (c_r > -0.2) & (c_r < 1.2)
    clean_cr = c_r[mask]
    clean_ci = c_i[mask]
    
    # Identify the most unstable mode
    max_idx = np.argmax(clean_ci)

    print("Plotting spectrum...")

    # --- Visualization ---
    plt.figure(figsize=(8, 7))
    
    # Plot all filtered modes
    plt.scatter(clean_cr, clean_ci, color='royalblue', edgecolors='k', alpha=0.7, s=40)
    
    # Highlight the most unstable mode
    plt.scatter(clean_cr[max_idx], clean_ci[max_idx], color='crimson', edgecolors='k', 
                s=80, marker='*', zorder=5, label=f'Leading Mode\n$c_r$={clean_cr[max_idx]:.4f}, $c_i$={clean_ci[max_idx]:.4f}')

    # Reference lines
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Neutral Stability ($c_i = 0$)')
    plt.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='$U_{max}$')
    plt.axvline(0.0, color='gray', linestyle=':', alpha=0.5, label='$U_{wall}$')

    # Formatting
    plt.title(f'Orr-Sommerfeld Eigenspectrum\nPlane Poiseuille Flow ($Re={Re}$, $\\alpha={k_val}$)', fontsize=14, pad=15)
    plt.xlabel('Phase Speed ($c_r$)', fontsize=13)
    plt.ylabel('Growth Rate ($c_i$)', fontsize=13)
    
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1.0, 0.05)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    plot_c_spectrum()