import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
def solve_evp(N):
    """
    Solves the eigenvalue problem for the interaction of unstably stratified shear flow
    and a phase boundary (melting solid) using Dedalus v3.
    """
    # 1. Wavenumber & Reynolds Scaling
    # The equations in the reference are non-dimensionalized by the full liquid depth h0.
    # To simulate classical textbook critical points (which use half-width), we scale:
    
    
    k = 2.0       # k_code = 2 * alpha_classic
    Pe = 100        # Pe_code = 2 * Pe_classic
    Pr = 7.0                      # Prandtl number
    Ra = 1e4                      # Rayleigh number
    Lambda = 1.0                  # Ratio of temperature differences
    S = 1.0                       # Stefan number
    d0 = Lambda                   # Initial solid thickness
    
    logger.info(f"Setting up EVP at N={N} with scaled Pe={Pe}, k={k}")
    coord = d3.Coordinate('z')
    dist = d3.Distributor(coord, dtype=np.complex128)
    basis_l = d3.Chebyshev(coord, size=N, bounds=(0, 1))
    basis_s = d3.Chebyshev(coord, size=N, bounds=(1, 1+d0))
    # Define Fields on explicit domains
    u = dist.Field(name='u', bases=basis_l)
    w = dist.Field(name='w', bases=basis_l)
    p = dist.Field(name='p', bases=basis_l)
    tl = dist.Field(name='tl', bases=basis_l)
    ts = dist.Field(name='ts', bases=basis_s)
    
    # Interface perturbation (global scalar)
    eta = dist.Field(name='eta')
    # 3. Eigenvalue Scaling
    # Solve directly for physical phase speed c (sigma = k * Pe * c) to prevent dense matrix failure.
    c = dist.Field(name='c')
    # Boundary lifting taus
    tau_p = dist.Field(name='tau_p')
    tau_u1 = dist.Field(name='tau_u1')
    tau_u2 = dist.Field(name='tau_u2')
    tau_w1 = dist.Field(name='tau_w1')
    tau_w2 = dist.Field(name='tau_w2')
    tau_tl1 = dist.Field(name='tau_tl1')
    tau_tl2 = dist.Field(name='tau_tl2')
    tau_ts1 = dist.Field(name='tau_ts1')
    tau_ts2 = dist.Field(name='tau_ts2')
    # 4. Dynamic Base Flows (No Hardcoding)
    # Define fields to evaluate unperturbed profiles and their derivatives dynamically.
    z_l = dist.local_grid(basis_l)
    z_s = dist.local_grid(basis_s)
    U = dist.Field(name='U', bases=basis_l)
    U['g'] = 4 * z_l * (1 - z_l)  # Parabolic profile
    Uz = d3.Differentiate(U, coord)
    Theta_l = dist.Field(name='Theta_l', bases=basis_l)
    Theta_l['g'] = 1 - z_l
    Theta_lz = d3.Differentiate(Theta_l, coord)
    Theta_s = dist.Field(name='Theta_s', bases=basis_s)
    Theta_s['g'] = (Lambda / d0) * (1 - z_s)
    Theta_sz = d3.Differentiate(Theta_s, coord)
    # Operators
    dz = lambda A: d3.Differentiate(A, coord)
    lift_basis_l = basis_l.derivative_basis(2)
    lift_l = lambda A, n: d3.Lift(A, lift_basis_l, n)
    lift_basis_s = basis_s.derivative_basis(2)
    lift_s = lambda A, n: d3.Lift(A, lift_basis_s, n)
    # Problem Setup
    problem = d3.EVP([u, w, p, tl, ts, eta, tau_p, 
                      tau_u1, tau_u2, tau_w1, tau_w2, tau_tl1, tau_tl2, tau_ts1, tau_ts2], 
                     eigenvalue=c, namespace=locals())
    # Liquid Equations (0 < z < 1)
    problem.add_equation("1j*k*u + dz(w) + tau_p = 0")
    problem.add_equation("-1j*k*Pe*c*u + Pe*(1j*k*U*u + w*Uz) + 1j*k*p - Pr*(dz(dz(u)) - k**2*u) + lift_l(tau_u1, -1) + lift_l(tau_u2, -2) = 0")
    problem.add_equation("-1j*k*Pe*c*w + Pe*(1j*k*U*w) + dz(p) - Ra*Pr/Pe*tl - Pr*(dz(dz(w)) - k**2*w) + lift_l(tau_w1, -1) + lift_l(tau_w2, -2) = 0")
    problem.add_equation("-1j*k*Pe*c*tl + Pe*(1j*k*U*tl + w*Theta_lz) - (dz(dz(tl)) - k**2*tl) + lift_l(tau_tl1, -1) + lift_l(tau_tl2, -2) = 0")
    
    # Solid Equations (1 < z < 1+d0)
    problem.add_equation("-1j*k*Pe*c*ts - (dz(dz(ts)) - k**2*ts) + lift_s(tau_ts1, -1) + lift_s(tau_ts2, -2) = 0")
    # Bottom Boundary (z=0)
    problem.add_equation("u(z=0) = 0")
    problem.add_equation("w(z=0) = 0")
    problem.add_equation("tl(z=0) = 0")
    # Top Boundary (z=1+d0)
    problem.add_equation("ts(z=1+d0) = 0")
    # Phase Boundary (z=1)
    problem.add_equation("u(z=1) + eta*Uz(z=1) = 0")
    problem.add_equation("w(z=1) = 0")
    problem.add_equation("tl(z=1) + eta*Theta_lz(z=1) = 0")
    problem.add_equation("ts(z=1) + eta*Theta_sz(z=1) = 0")
    # 5. Strict Sign Convention (Stefan Condition)
    problem.add_equation("-1j*k*Pe*c*eta + 1/(Lambda*S) * (dz(tl)(z=1) - dz(ts)(z=1)) = 0")
    # Pressure Gauge (replaces continuity zero-mode nullspace)
    problem.add_equation("integ(p) = 0")
    # Solve dense
    solver = problem.build_solver()
    solver.solve_dense(solver.subproblems[0])
    
    return solver.eigenvalues
def filter_and_plot():
    """
    6. Spurious Mode Filtering
    Solves at two grid resolutions and filters tau-modes before plotting.
    """
    N1 = 160
    N2 = 200
    k=1
    Pe=100
    logger.info("Solving EVP at N=160...")
    evals1 = solve_evp(N1)
    
    logger.info("Solving EVP at N=200...")
    evals2 = solve_evp(N2)
    
    logger.info("Filtering spurious modes (threshold < 1e-5)...")
    threshold = 1e-5
    good_evals = []
    
    for ev1 in evals1:
        # Distance to nearest eigenvalue in high-res solve
        dist = np.abs(evals2 - ev1)
        nearest_idx = np.argmin(dist)
        ev2 = evals2[nearest_idx]
        
        # Check relative change metric (Delta lambda / |lambda|)
        rel_change = np.abs(ev1 - ev2) / np.abs(ev1) if np.abs(ev1) > 1e-10 else np.abs(ev1 - ev2)
            
        if rel_change < threshold:
            good_evals.append(ev1)
            
    good_evals = np.array(good_evals)
    
    logger.info(f"Converged physical modes: {len(good_evals)} / {len(evals1)}")
    
    # Plot Phase Speed c
    plt.figure(figsize=(8, 6))
    plt.scatter(np.real(good_evals)/(k*Pe), np.imag(good_evals)/(k*Pe), alpha=0.7, edgecolors='k', s=40)
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.title('Filtered Phase Speed Spectrum (Dedalus v3)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(0, color='red', linestyle='-', alpha=0.5)
    plt.ylim(-5,1)
    plt.xlim(-1,5)
    # Save the plot
    plt.savefig('spectra_c.png', dpi=300, bbox_inches='tight')
    logger.info("Saved spectrum to 'spectra_c.png'")
    
    # Uncomment plt.show() below if you want the interactive window to pop up:
    # plt.show()
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filter_and_plot()