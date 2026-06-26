import numpy as np
import dedalus.public as de
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 1. Parameters (From your latest code)
# -------------------------------------------------------------------------
Ra = 0     # Rayleigh number
Pe = 4000.0      # Péclet number 
Pr = 7.0         # Prandtl number
S = 1          # Stefan number
Lambda = 1     # Thermal conductivity ratio
d0 = Lambda      # Solid layer thickness (from Image 3)
k = 1.0          # Streamwise wavenumber
m = 0.0          # Spanwise wavenumber
gamma2 = k**2 + m**2

# -------------------------------------------------------------------------
# 2. Encapsulated Solver Function (To enable filtering)
# -------------------------------------------------------------------------
def solve_spectrum(Nz):
    """Builds and solves the exact Generalized EVP from the paper for a given grid size."""
    zcoord = de.Coordinate('z')
    dist = de.Distributor(zcoord, dtype=np.complex128)
    
    # Unified code grid: z=1 is the interface, z=0 is the outer walls
    zb = de.ChebyshevT(zcoord, size=Nz, bounds=(0, 1))
    z = dist.local_grid(zb)

    # Base State Velocity (Image 3: u^(0) = 1 - z)
    U = dist.Field(name='U', bases=zb)
    U['g'] = 1 - z

    # State Variables
    w = dist.Field(name='w', bases=zb)
    wz = dist.Field(name='wz', bases=zb)
    wzz = dist.Field(name='wzz', bases=zb)
    wzzz = dist.Field(name='wzzz', bases=zb)
    Tl = dist.Field(name='Tl', bases=zb)
    Tlz = dist.Field(name='Tlz', bases=zb)
    Ts = dist.Field(name='Ts', bases=zb)
    Tsz = dist.Field(name='Tsz', bases=zb)
    h = dist.Field(name='h')
    sigma = dist.Field(name='sigma')

    # Tau fields for boundary enforcement
    tau_w1 = dist.Field(name='tau_w1')
    tau_w2 = dist.Field(name='tau_w2')
    tau_w3 = dist.Field(name='tau_w3')
    tau_w4 = dist.Field(name='tau_w4')
    tau_Tl1 = dist.Field(name='tau_Tl1')
    tau_Tl2 = dist.Field(name='tau_Tl2')
    tau_Ts1 = dist.Field(name='tau_Ts1')
    tau_Ts2 = dist.Field(name='tau_Ts2')

    D = lambda A: de.Differentiate(A, zcoord)
    lift_basis = zb.derivative_basis(1)
    lift = lambda A: de.Lift(A, lift_basis, -1)

    variables = [w, wz, wzz, wzzz, Tl, Tlz, Ts, Tsz, h,
                 tau_w1, tau_w2, tau_w3, tau_w4, 
                 tau_Tl1, tau_Tl2, tau_Ts1, tau_Ts2]
    problem = de.EVP(variables, eigenvalue=sigma)

    # First-order reduction definitions
    problem.add_equation((D(w) - wz + lift(tau_w1), 0))
    problem.add_equation((D(wz) - wzz + lift(tau_w2), 0))
    problem.add_equation((D(wzz) - wzzz + lift(tau_w3), 0))
    problem.add_equation((D(Tl) - Tlz + lift(tau_Tl1), 0))
    problem.add_equation((D(Ts) - Tsz + lift(tau_Ts1), 0))

    # -------------------------------------------------------------------------
    # EQUATIONS (Mapped exactly from Image 4 to LHS - RHS = 0)
    # -------------------------------------------------------------------------
    
    # 1. Fluid Vertical Velocity 
    eq_os = ( -1j * sigma * (wzz - gamma2 * w) 
              - Pr * (D(wzzz) - 2 * gamma2 * wzz + gamma2**2 * w) 
              + 1j * k * Pe * U * (wzz - gamma2 * w) 
              + gamma2 * (Ra * Pr / Pe) * Tl 
              + lift(tau_w4) )
    problem.add_equation((eq_os, 0))

    # 2. Liquid Temperature (Note: Pe * w comes from Pe(Dθ_l^(0))w where Dθ_l^(0) = -1)
    eq_Tl = ( -1j * sigma * Tl 
              - (D(Tlz) - gamma2 * Tl) 
              + 1j * k * Pe * U * Tl 
              + Pe * w 
              + lift(tau_Tl2) )
    problem.add_equation((eq_Tl, 0))

    # 3. Solid Temperature (Mapped to code domain: D_phys^2 = (1/d0^2) * D_code^2)
    eq_Ts = ( -1j * sigma * Ts 
              - (1/d0**2) * D(Tsz)   #-change
              + gamma2 * Ts 
              + lift(tau_Ts2) )
    problem.add_equation((eq_Ts, 0))

    # 4. Stefan Condition (Evaluated at z=1)
    # Image: -1/(Lambda*S) * D_phys(Tl) + 1/(Lambda*S) * D_phys(Ts) = -i*sigma*h
    # Mapping: D_phys(Ts) = -(1/d0) * D_code(Ts)
    eq_stefan = ( -1j * sigma * h 
                  + (1 / (Lambda * S)) * Tlz(z=1) 
                  - (1 / (Lambda * S * d0)) * Tsz(z=1) )
    problem.add_equation((eq_stefan, 0))

    # -------------------------------------------------------------------------
    # BOUNDARY CONDITIONS (Mapped exactly from Image 2)
    # -------------------------------------------------------------------------
    
    # At the bottom wall (z=0)
    problem.add_equation((w(z=0), 0))
    problem.add_equation((wz(z=0), 0))
    problem.add_equation((Tl(z=0), 0))

    # At the top boundary of the solid (Physical z=1+d0 maps to Code z=0)
    problem.add_equation((Ts(z=0), 0))

    # At the moving interface (z=1)
    problem.add_equation((w(z=1), 0))
    problem.add_equation((wz(z=1) + 1j * k * h, 0))         # Dw - ik(-1)h = 0
    problem.add_equation((Tl(z=1) - h, 0))                  # Tl + (-1)h = 0
    problem.add_equation((Ts(z=1) - (Lambda/d0) * h, 0))    # Ts + (-Lambda/d0)h = 0 (From Image 3)

    # Solve dense matrix
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    
    evals = solver.eigenvalues
    return evals[np.isfinite(evals)]

# -------------------------------------------------------------------------
# 3. Two-Resolution Intersection Filter
# -------------------------------------------------------------------------
logger.info("Solving at Nz=100 and Nz=120 to filter spurious numerical modes...")
evals1 = solve_spectrum(100)
evals2 = solve_spectrum(120)

physical_modes = []
tolerance = 1e-4

for e1 in evals1:
    # If the eigenvalue doesn't change when resolution changes, it is real physics.
    min_dist = np.min(np.abs(evals2 - e1))
    if min_dist < tolerance:
        physical_modes.append(e1)

physical_modes = np.array(physical_modes)

# Map sigma to standard phase speed: sigma = k * c
growth_rates = physical_modes.imag / (k*Pe)
phase_speeds = physical_modes.real / (k*Pe)

unstable_idx = np.argmax(growth_rates)
c_r_unstable = phase_speeds[unstable_idx]
c_i_unstable = growth_rates[unstable_idx]

logger.info(f"Filtered down to {len(physical_modes)} physical modes.")
logger.info(f"Most Unstable Mode: c_r = {c_r_unstable:.6f}, c_i = {c_i_unstable:.6f}")

# -------------------------------------------------------------------------
# 4. Plotting the Clean Spectrum
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 8))

# Plot physical modes
plt.scatter(phase_speeds, growth_rates, edgecolors='blue', facecolors='none', 
            s=60, label='Physical Eigenmodes', zorder=2)

# Highlight most unstable mode
plt.scatter(c_r_unstable, c_i_unstable, color='gold', marker='*', 
            s=400, edgecolor='black', label='Most Unstable Mode', zorder=4)

plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=1)
plt.axvline(0, color='gray', linestyle='--', linewidth=1.5, zorder=1)

# Dynamically frame the plot based on Pe
plt.xlim(-1,2)
plt.ylim(-3,1) 

plt.xlabel(r"Phase Speed ($c_r$)", fontsize=14)
plt.ylabel(r"Growth Rate ($c_i$)", fontsize=14)
plt.title(f"Coupled Phase-Change Spectrum (Cleaned)\nRa={Ra}, Pe={Pe}, k={k}", fontsize=16)

plt.axhspan(0, max(growth_rates) + 50, facecolor='red', alpha=0.05, zorder=0)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()
plt.show()
