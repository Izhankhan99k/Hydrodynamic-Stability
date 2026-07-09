import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de
import logging

# Mute Dedalus solver logs to keep output clean during two runs
logger = logging.getLogger('evaluator')
logger.setLevel(logging.WARNING)
de_logger = logging.getLogger('pencil')
de_logger.setLevel(logging.WARNING)

# -------------------------------------------------------------------------
# Physical Parameters
# -------------------------------------------------------------------------
Pr = 10.0       
Ra = 1e4        
Pe = 50.0       
S = 1.5         
Lambda = 0.5    
d0 = 0.5        
k = 2.0         

# -------------------------------------------------------------------------
# The Solver Function
# -------------------------------------------------------------------------
def solve_evp(Nz_liquid, Nz_solid):
    """Builds and solves the EVP for a given resolution."""
    print(f"Solving at resolution: Liquid={Nz_liquid}, Solid={Nz_solid}...")
    
    # 1. Domain Setup
    z_liquid = de.Chebyshev('z', Nz_liquid, interval=(0, 1))
    z_solid = de.Chebyshev('z', Nz_solid, interval=(1, 1 + d0))
    domain = de.Domain([z_liquid, z_solid], grid_dtype=np.complex128)

    # 2. Problem Formulation
    variables = ['w', 'wz', 'zeta', 'zetaz', 'theta_l', 'theta_lz', 'theta_s', 'theta_sz']
    problem = de.EVP(domain, variables=variables, eigenvalue='sigma')

    # Masks
    z = domain.grid(0)
    L_mask = domain.new_field()
    L_mask['g'] = (z <= 1.0).astype(np.float64)
    S_mask = domain.new_field()
    S_mask['g'] = (z > 1.0).astype(np.float64)

    problem.parameters['L'] = L_mask
    problem.parameters['S'] = S_mask
    problem.parameters['k'] = k
    problem.parameters['Pr'] = Pr
    problem.parameters['Ra'] = Ra
    problem.parameters['Pe'] = Pe
    problem.parameters['Stefan'] = S      
    problem.parameters['Lambda'] = Lambda

    # Base flow (Only in liquid)
    U = domain.new_field()
    U['g'] = 4 * z * (1 - z) * L_mask['g']
    problem.parameters['U'] = U

    # 3. Governing Equations
    problem.add_equation("L*(dz(w) - wz) + S*(w) = 0")
    problem.add_equation("L*(dz(wz) - k**2 * w - zeta) + S*(wz) = 0")
    problem.add_equation("L*(dz(zeta) - zetaz) + S*(zeta) = 0")
    
    # Modified Orr-Sommerfeld (with the corrected -8 sign)
    problem.add_equation("L*(Pr*(dz(zetaz) - k**2 * zeta) - sigma*zeta - 1j*k*Pe*U*zeta - 8*1j*k*Pe*w - k**2*(Ra*Pr/Pe)*theta_l) + S*(zetaz) = 0")
    
    # Energy Equations
    problem.add_equation("L*(dz(theta_lz) - k**2 * theta_l - sigma*theta_l - 1j*k*Pe*U*theta_l + Pe*w) + S*(theta_l) = 0")
    problem.add_equation("S*(dz(theta_sz) - k**2 * theta_s - sigma*theta_s) + L*(theta_s) = 0")

    # 4. Boundary Conditions
    problem.add_bc("left(w) = 0")
    problem.add_bc("left(wz) = 0")
    problem.add_bc("left(theta_l) = 0")
    problem.add_bc("interp(w, z=1) = 0")
    problem.add_bc("interp(wz, z=1) + 4*1j*k*interp(theta_l, z=1) = 0")                
    problem.add_bc("interp(theta_l, z=1) - interp(theta_s, z=1) = 0")                   
    problem.add_bc("sigma*interp(theta_l, z=1) - (1/(Lambda*Stefan))*(interp(theta_sz, z=1) - interp(theta_lz, z=1)) = 0") 
    problem.add_bc("right(theta_s) = 0")

    # 5. Solve
    solver = problem.build_solver()
    solver.solve_dense(solver.pencils[0])
    
    # Return finite eigenvalues
    evals = solver.eigenvalues
    return evals[np.isfinite(evals)]


# -------------------------------------------------------------------------
# The Two-Resolution Routine
# -------------------------------------------------------------------------
# Base resolution
N_L1, N_S1 = 64, 32
evals_base = solve_evp(N_L1, N_S1)

# High resolution (~1.5x)
N_L2, N_S2 = 96, 48
evals_high = solve_evp(N_L2, N_S2)

# Cross-matching to filter spurious modes
tolerance = 1e-4  # Maximum allowed drift in the complex plane
physical_evals = []

for e1 in evals_base:
    # Calculate the distance from this base eigenvalue to ALL high-res eigenvalues
    drift = np.min(np.abs(evals_high - e1))
    
    # If the closest high-res eigenvalue is within tolerance, it is a physical mode
    if drift < tolerance:
        physical_evals.append(e1)

physical_evals = np.array(physical_evals)

print(f"Total modes at base resolution: {len(evals_base)}")
print(f"Physical modes retained: {len(physical_evals)}")

if len(physical_evals) > 0:
    max_growth = np.max(physical_evals.real)
    print(f"Most unstable physical mode (sigma_r): {max_growth:.6f}")
else:
    print("No physical modes found within the given tolerance.")

# -------------------------------------------------------------------------
# Plotting the Cleaned Spectrum
# -------------------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Plot physical modes
plt.scatter(physical_evals.real, physical_evals.imag, marker='o', alpha=0.9, 
            color='blue', edgecolors='k', s=40, label='Physical Modes')

# Highlight unstable modes (if any)
unstable = physical_evals[physical_evals.real > 0]
if len(unstable) > 0:
    plt.scatter(unstable.real, unstable.imag, color='red', marker='o', 
                s=80, edgecolors='k', label='Unstable Physical Modes')

plt.axvline(0, color='red', linestyle='--', label='Stability Boundary')
plt.axhline(0, color='black', linewidth=0.8)
plt.title(f'Cleaned Eigenvalue Spectrum ($k={k}$, $Ra={Ra:.1e}$)', fontsize=14)
plt.xlabel('Growth Rate, $\sigma_r$ (Real Part)', fontsize=12)
plt.ylabel('Phase Speed, $\sigma_i$ (Imaginary Part)', fontsize=12)

# Dynamically scale axis to focus on physical modes
if len(physical_evals) > 0:
    plt.xlim([np.min(physical_evals.real) - 10, np.max(physical_evals.real) + 10])
    plt.ylim([np.min(physical_evals.imag) - 50, np.max(physical_evals.imag) + 50])

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()