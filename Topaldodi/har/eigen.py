import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
import logging

# Mute noisy Dedalus logs for cleaner output
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('evaluator').setLevel(logging.WARNING)
logging.getLogger('matrix').setLevel(logging.WARNING)
logging.getLogger('problems').setLevel(logging.WARNING)

def solve_evp(k_val, a_val, Re_val, G_val, Bo_val, nz):
    """
    Builds and solves the Orr-Sommerfeld EVP for a specific resolution `nz`.
    Returns the raw array of complex phase speeds (c).
    """
    zcoord = de.Coordinate('z')
    dist = de.Distributor(zcoord, dtype=np.complex128)
    zbasis = de.Chebyshev(zcoord, size=nz, bounds=(-1, 0))
    
    z_grid = dist.local_grid(zbasis)
    k = k_val
    Re = Re_val
    G = G_val
    Bo = Bo_val
    a = a_val
    
    dz = lambda A: de.Differentiate(A, zcoord)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: de.Lift(A, lift_basis, -1)
    
    phi = dist.Field(name='phi', bases=zbasis)
    phi_z = dist.Field(name='phi_z', bases=zbasis)
    phi_zz = dist.Field(name='phi_zz', bases=zbasis)
    phi_zzz = dist.Field(name='phi_zzz', bases=zbasis)
    
    eta = dist.Field(name='eta')
    sigma = dist.Field(name='sigma')
    
    tau_1 = dist.Field(name='tau_1')
    tau_2 = dist.Field(name='tau_2')
    tau_3 = dist.Field(name='tau_3')
    tau_4 = dist.Field(name='tau_4')
    
    U = dist.Field(name='U', bases=zbasis)
    Uz = dist.Field(name='Uz', bases=zbasis)
    Uzz = dist.Field(name='Uzz', bases=zbasis)
    
    U['g'] = a * z_grid**2 + (a + 1) * z_grid + 1
    Uz['g'] = 2 * a * z_grid + (a + 1)
    Uzz['g'] = 2 * a
    
    variables = [phi, phi_z, phi_zz, phi_zzz, eta, tau_1, tau_2, tau_3, tau_4]
    problem = de.EVP(variables, eigenvalue=sigma, namespace=locals())
    
    problem.add_equation("dz(phi) - phi_z + lift(tau_1) = 0")
    problem.add_equation("dz(phi_z) - phi_zz + lift(tau_2) = 0")
    problem.add_equation("dz(phi_zz) - phi_zzz + lift(tau_3) = 0")
    
    problem.add_equation("sigma*(phi_zz - k**2*phi) + 1j*k*U*(phi_zz - k**2*phi) - 1j*k*Uzz*phi - (1/Re)*(dz(phi_zzz) - 2*k**2*phi_zz + k**4*phi) + lift(tau_4) = 0")
    
    problem.add_equation("phi(z=-1) = 0")
    problem.add_equation("phi_z(z=-1) = 0")
    
    problem.add_equation("eta*Uzz(z=0) + phi_zz(z=0) + k**2*phi(z=0) = 0")
    problem.add_equation("sigma*eta + 1j*k*U(z=0)*eta + 1j*k*phi(z=0) = 0")
    problem.add_equation("sigma*Re*phi_z(z=0) - phi_zzz(z=0) + 3*k**2*phi_z(z=0) - 1j*k*Re*Uz(z=0)*phi(z=0) + 1j*k*Re*U(z=0)*phi_z(z=0) + 1j*k*Re*G*(1 + k**2/Bo)*eta = 0")
    
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
    
    # Return phase speeds (c = i * sigma / k)
    return 1j * solver.eigenvalues / k


# =========================================================
# Execution & Plotting the Spectrum
# =========================================================
if __name__ == "__main__":
    
    # Pick a single slice of parameter space that we know is interesting
    k_test = 1.0
    Re_test = 1000.0
    a_convex = 0.05
    
    # Solve at two different resolutions
    N1 = 120
    N2 = 140
    
    print(f"Solving Matrix at N={N1}...")
    c1 = solve_evp(k_test, a_convex, Re_test, 0.0, 1.0, N1)
    
    print(f"Solving Matrix at N={N2}...")
    c2 = solve_evp(k_test, a_convex, Re_test, 0.0, 1.0, N2)
    
    # Clean NaNs and Infs
    c1 = c1[np.isfinite(c1)]
    c2 = c2[np.isfinite(c2)]
    
    # Find the strictly physical modes that overlap
    tol = 1e-5
    physical_c = []
    for c in c1:
        if np.min(np.abs(c2 - c)) < tol:
            physical_c.append(c)
    physical_c = np.array(physical_c)

    print("Plotting Eigenspectrum...")

    # Plotting
    plt.figure(figsize=(10, 7))
    
    # 1. Plot the raw outputs from both grids
    plt.scatter(c1.real, c1.imag, facecolors='none', edgecolors='blue', 
                s=80, alpha=0.7, label=f'Raw Output ($N={N1}$)')
                
    plt.scatter(c2.real, c2.imag, marker='+', color='red', 
                s=50, alpha=0.7, label=f'Raw Output ($N={N2}$)')
                
    # 2. Highlight the filtered physical modes
    plt.scatter(physical_c.real, physical_c.imag, color='black', 
                s=15, zorder=5, label='Converged Physical Modes')

    # Draw the Neutral Stability Threshold (c_i = 0)
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Neutral Stability ($c_i = 0$)')
    
    # Because spurious modes can have wave speeds of 10,000, 
    # we heavily restrict the axes to the physical window where the fluid actually exists.
    # The fluid velocity is bounded between 0 and U_max (~1.0).
    plt.xlim(-0.5, 1000)
    plt.ylim(-1100, 10)
    
    # Highlight the unstable region
    plt.axhspan(0, 0.5, color='red', alpha=0.05, label='Unstable ($\sigma_r > 0$)')

    # Formatting
    plt.title(f'Complex Eigenspectrum\n($k = {k_test}, Re = {Re_test}, a = {a_convex}$)', fontsize=14, pad=10)
    plt.xlabel('Real Phase Speed ($c_r$)', fontsize=12)
    plt.ylabel('Imaginary Phase Speed / Growth Rate ($c_i$)', fontsize=12)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10, framealpha=1.0)
    
    plt.tight_layout()
    plt.show()