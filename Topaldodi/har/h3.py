import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
import logging
import sys

# Mute noisy Dedalus logs for cleaner output
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('evaluator').setLevel(logging.WARNING)
logging.getLogger('matrix').setLevel(logging.WARNING)
logging.getLogger('problems').setLevel(logging.WARNING)

def solve_evp(k_val, a_val, Re_val, G_val, Bo_val, nz):
    """
    Builds and solves the Orr-Sommerfeld EVP for a specific resolution `nz`.
    Returns the raw array of eigenvalues.
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
    
    return solver.eigenvalues


def get_max_ci_two_res(k, a, Re=10.0, G=0.0, Bo=1.0, N1=64, N2=96, tol=1e-5):
    """
    Solves the EVP at two different grid resolutions. 
    Cross-references the results to filter out spurious eigenvalues.
    """
    # 1. Solve at both resolutions
    evals1 = solve_evp(k, a, Re, G, Bo, N1)
    evals2 = solve_evp(k, a, Re, G, Bo, N2)
    
    # 2. Convert to phase speeds: sigma = -1j * k * c  =>  c = 1j * sigma / k
    c1 = 1j * evals1 / k
    c2 = 1j * evals2 / k
    
    # Clean up NaNs and Infs
    c1 = c1[np.isfinite(c1)]
    c2 = c2[np.isfinite(c2)]
    
    physical_ci = []
    
    # 3. The Two-Resolution Cross-Check
    for c in c1:
        # Calculate the distance from this root to ALL roots in the higher resolution
        dist = np.min(np.abs(c2 - c))
        
        # If the root hasn't moved (within tolerance), it is a true physical mode
        if dist < tol:
            physical_ci.append(c.imag)
            
    # 4. Extract the most unstable mode
    if len(physical_ci) > 0:
        max_ci = np.max(physical_ci)
        
        # We only return strictly unstable modes for plotting on the log scale
        if max_ci > 1e-10:
            return max_ci
            
    return np.nan


# =========================================================
# Execution & Plotting
# =========================================================
if __name__ == "__main__":
    
    # Setup wavenumber array (log space from 10^-2 to 10^1)
    k_array = np.logspace(-2, 1, 60)
    
    a_convex = 0.05
    a_concave = -0.05
    
    ci_convex = []
    ci_concave = []

    print("Executing Two-Resolution Filter... this will take slightly longer due to double-solves.")
    for k in k_array:
        sys.stdout.write('.')
        sys.stdout.flush()
        # The filter automatically discards spurious modes
        ci_convex.append(get_max_ci_two_res(k, a_convex))
        ci_concave.append(get_max_ci_two_res(k, a_concave))
        
    print("\nSolving complete. Generating plot...")

    # Plotting to perfectly match the paper's style
    plt.figure(figsize=(8, 5))
    
    plt.loglog(k_array, ci_convex, label=r'$a = 0.05$, numerical calculation', color='#0072BD', linewidth=2)
    plt.loglog(k_array, ci_concave, label=r'$a = -0.05$, numerical calculation', color='#EDB120', linewidth=2)

    plt.title(r'$Re = 10, G = 0$', pad=10)
    plt.xlabel(r'$k$', fontsize=12)
    plt.ylabel(r'$c_i$', fontsize=12, rotation=0, labelpad=15)
    
    # Set matching axis limits from the image
    plt.xlim(10**-2, 10**1)
    plt.ylim(10**-6, 10**0)
    
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    
    plt.legend(loc='lower left', framealpha=1.0, edgecolor='black')
    plt.tight_layout()
    plt.show()