import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

# Suppress logging
logging.getLogger('dedalus').setLevel(logging.WARNING)

def plot_c_spectrum_mapped():
    # Textbook parameters for the classic "Y-shaped" Poiseuille spectrum
    Re = 10000.0
    k_val = 1.0
    j = 1j
    N = 128  

    print(f"Solving Mapped OS EVP [0, 1] for Re={Re}, k={k_val}, N={N}...")

    # Domain: [0, 1]
    zcoord = d3.Coordinate('z')
    dist = d3.Distributor(zcoord, dtype=np.complex128)
    basis = d3.ChebyshevT(zcoord, size=N, bounds=(0, 1))
    z = dist.local_grid(basis)

    # Base State: Mapped to [0, 1], but derivatives scaled to half-width
    U = dist.Field(name='U', bases=basis)
    U['g'] = 4*z*(1-z)         # Max velocity is 1.0 at z=0.5
    Uyy=dist.Field(name='U',bases=basis)
    Uyy['g']=-2*np.ones_like(z)
       

    # Fields & Tau Polynomials (Representing y-derivatives)
    w = dist.Field(name='w', bases=basis)
    wy = dist.Field(name='wy', bases=basis)
    wyy = dist.Field(name='wyy', bases=basis)
    wyyy = dist.Field(name='wyyy', bases=basis)
 

    sigma = dist.Field(name='sigma')

    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')
    tau_3 = dist.Field(name='tau_3')
    tau_4 = dist.Field(name='tau_4')
   


    # THE CHAIN RULE OPERATORS
    dz = lambda A: d3.Differentiate(A, zcoord)
    dy = lambda A: 0.5 * dz(A)   # Scales back to the physical [-1, 1] length
    lift = lambda A: d3.Lift(A, basis.derivative_basis(1), -1)

    problem = d3.EVP([w, wy, wyy, wyyy ,tau_1, tau_2, tau_3, tau_4], eigenvalue=sigma, namespace=locals())

    # Build equations using 'dy' and 'wyy'
    problem.add_equation("dy(w) - wy + lift(tau_1) = 0")
    problem.add_equation("dy(wy) - wyy + lift(tau_2) = 0")
    problem.add_equation("dy(wyy) - wyyy + lift(tau_3) = 0")
   
    problem.add_equation(
        "sigma*(wyy - k_val**2 * w) "
        "+ j*k_val*U*(wyy - k_val**2 * w) - j*k_val*Uyy*w "
        "- (1/Re)*(dy(wyyy) - 2*k_val**2 * wyy + k_val**4 * w) + lift(tau_4) = 0"
    )

    # Boundary Conditions (Walls are now at z=0 and z=1)
    problem.add_equation("w(z=0) = 0")
    problem.add_equation("wy(z=0) = 0")
    problem.add_equation("w(z=1) = 0")
    problem.add_equation("wy(z=1) = 0")

    # Solve
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])

    # Extract and convert to Wave Speed
    evals = solver.eigenvalues
    evals = evals[np.isfinite(evals)]

    c_r = -evals.imag / k_val
    c_i = evals.real / k_val

    # Filter out numerical artifacts
    mask = (c_i > -1.0) & (c_r > -0.2) & (c_r < 1.2)
    clean_cr = c_r[mask]
    clean_ci = c_i[mask]
    
    max_idx = np.argmax(clean_ci)

    print("Plotting spectrum...")

    # Visualization
    plt.figure(figsize=(8, 7))
    plt.scatter(clean_cr, clean_ci, color='royalblue', edgecolors='k', alpha=0.7, s=40)
    plt.scatter(clean_cr[max_idx], clean_ci[max_idx], color='crimson', edgecolors='k', 
                s=80, marker='*', zorder=5, label=f'Leading Mode\n$c_r$={clean_cr[max_idx]:.4f}, $c_i$={clean_ci[max_idx]:.4f}')

    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Neutral Stability ($c_i = 0$)')
    plt.axvline(1.0, color='gray', linestyle=':', alpha=0.5, label='$U_{max}$')
    plt.axvline(0.0, color='gray', linestyle=':', alpha=0.5, label='$U_{wall}$')

    plt.title(f'Mapped Orr-Sommerfeld Eigenspectrum [0, 1]\nPlane Poiseuille Flow ($Re={Re}$, $\\alpha={k_val}$)', fontsize=14, pad=15)
    plt.xlabel('Phase Speed ($c_r$)', fontsize=13)
    plt.ylabel('Growth Rate ($c_i$)', fontsize=13)
    
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1.0, 0.05)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    plot_c_spectrum_mapped()