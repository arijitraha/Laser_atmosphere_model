import numpy as np
from numba import njit
from scipy.optimize import root
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Laser & Atmospheric Dynamics", layout="wide")
st.title("Laser and Atmospheric Dynamical Systems Analysis")
@njit
def equations(vars, N0, gamma, A, a):
  N, I = vars
  f = gamma * (N0 - N - N * I)
  g = I * (N - 1 - A / (1 + a * I))
  return f, g

@st.cache_data
def fixed_points(N0, A, a):
    found_points = []
    b = 1 + a + A - a * N0
    c = 1 + A - N0
    if a == 0 and A == 0:
        I_star = N0 - 1
    else:
        I_star = (-b + np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
    N_star = N0 / (1 + I_star) if I_star != -1 else N0
    if N_star >= 0 and I_star >= 0:
        # Check if this fixed point is already found (deduplication)
        is_new = True
        for fp in found_points:
            if np.hypot(fp[0] - N_star, fp[1] - I_star) < 1e-3:
                is_new = False
                break
        if is_new:
            found_points.append((N_star, I_star))

    return found_points

@njit(fastmath=True)
def solve_rk4(a, H_init, R_init, H0, gamma2, A, t_max, dt):
  n_steps = int(t_max / dt)
  t = np.linspace(0, t_max, n_steps)
  H = np.zeros(n_steps)
  R = np.zeros(n_steps)

  H[0], R[0] = H_init, R_init
  for i in range(n_steps - 1):
    h, r = H[i], R[i]

    # Inline RK4 Butcher tableau step calculations
    k1_H, k1_R = equations((h, r), H0, gamma2, A, a)
    k2_H, k2_R = equations(
        (h + 0.5 * dt * k1_H, r + 0.5 * dt * k1_R), H0, gamma2, A, a
    )
    k3_H, k3_R = equations(
        (h + 0.5 * dt * k2_H, r + 0.5 * dt * k2_R), H0, gamma2, A, a
    )
    k4_H, k4_R = equations((h + dt * k3_H, r + dt * k3_R), H0, gamma2, A, a)

    H[i + 1] = h + (dt / 6.0) * (k1_H + 2.0 * k2_H + 2.0 * k3_H + k4_H)
    R[i + 1] = r + (dt / 6.0) * (k1_R + 2.0 * k2_R + 2.0 * k3_R + k4_R)

  return t, H, R

@st.cache_data
def run_bifurcation_sweep(a_array):
    R_stars, R_mins, R_maxs = [], [], []
    for a1 in a_array:
        pts = fixed_points(H0, A, a1)
        if len(pts) > 0:
            R_star = pts[0][1] if len(pts) > 0 else 0.0
            R_stars.append(R_star)
        # Check Jacobian trace for stability switch
        tr = -gamma2 * (1.0 + R_star) + (a1 * A * R_star) / ((1.0 + a1 * R_star)**2)

        if tr > 0:
            t, H, R = solve_rk4(a1, H_init=R_star + 0.2, R_init=R_star + 0.2, H0=H0, gamma2=gamma2, A=A, t_max=600, dt=0.01)
            tail_idx = int(0.7 * len(R))
            R_mins.append(np.min(R[tail_idx:]))
            R_maxs.append(np.max(R[tail_idx:]))
        else:
            R_mins.append(R_star)
            R_maxs.append(R_star)

    return R_stars, R_mins, R_maxs

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Laser System Settings (Standard 2D rate equation)")
    n0_val = st.slider("Normalised pump rate ($N_0$)", min_value=1.1, max_value=5.0, value=1.6, step=0.1)
    gamma_val = st.slider("Normalised gain coefficient ($\\gamma$)", min_value=0.05, max_value=1.0, value=0.2, step=0.01)
   
    with st.spinner("Buffering Laser computations..."):
        fixed_pts = fixed_points(n0_val, 0, 0)
        st.subheader("Laser System Analysis Results")
        for pt in fixed_pts:
            st.write(f"**Fixed point:** ({pt[0]:.3f}, {pt[1]:.3f})")
            J = np.array([[-gamma_val * (1 + pt[1]), -gamma_val * pt[0]], [pt[1], pt[0] - 1]])
            st.text(f"Jacobian matrix:\n{J}")
            eigenvals = np.linalg.eigvals(J)
            st.write(f"**Eigenvalues:** {eigenvals[0]:.3f} and {eigenvals[1]:.3f}")
            if np.all(np.real(eigenvals) < 0):
                st.info("As the eigenvalues are complex conjugate and the real part is negative, the system exhibits damped relaxation oscillations but cannot produce self sustaining limit cycles.")

with col2:
    st.header("Atmospheric Model Settings")
    H0 = 10.0
    gamma2 = 0.1
    A = 2.0
    st.write(f"**Maximum humidity capacity ($H_0$):** {H0}")
    st.write(f"**Moisture recharge rate ($\\gamma$):** {gamma2}")
    st.write(f"**Baseline aerosol microphysical suppression ($A$):** {A}")

    a_input = st.text_input("Enter cloud invigoration sensitivity ($a$) [0.01 to 0.5]:", value="0.18")
    try:
        a_float = float(a_input) if a_input.strip() else 0.18
        if not (0.01 <= a_float <= 0.5):
            st.error("Input out of bounds. Please enter a value between 0.01 and 0.5.")
            a_float = 0.18
    except ValueError:
        st.error("Invalid input detected. Defaulting cloud invigoration sensitivity (a) to 0.18")
        a_float = 0.18
    stable = True
    with st.spinner("Buffering Atmospheric computations..."):
        fixed_pts1 = fixed_points(H0, A, a_float)
        st.subheader("Atmospheric Steady State Analysis")
        for pt1 in fixed_pts1:
            st.write(f"**Fixed point for a = {a_float:.2f}:** ({pt1[0]:.3f}, {pt1[1]:.3f})")
            J1 = np.array([[-gamma2 * (1 + pt1[1]), -gamma2 * pt1[0]], [pt1[1], (A * a_float * pt1[1]) / (1 + a_float * pt1[1])**2]])
            trace_J1 = np.trace(J1)
            det_J1 = np.linalg.det(J1)
            st.text(f"Jacobian matrix:\n{J1}")
            st.write(f"**Trace of Jacobian:** {trace_J1:.3f} | **Determinant:** {det_J1:.3f}")
            if trace_J1 < 0 and det_J1 > 0:
                st.success("The trace is negative and the determinant is positive: the steady state is stable.")
                stable = True
            else:
                st.warning("The steady state is unstable.")
                stable = False
st.markdown("---")
st.header("Interactive Dynamics & Visualization")

with st.spinner("Calculating critical thresholds and plotting bifurcation sweep..."):
    a_val = np.linspace(0.01, 0.5, 10000)
    found_ac = []
    prev_trace = None
    prev_a1 = None
    prev_pt = None
    for a1 in a_val:
        fixed_pts2 = fixed_points(H0, A, a1)
        if len(fixed_pts2) > 0:
            pt2 = fixed_pts2[0]
            J2 = np.array([[-gamma2 * (1 + pt2[1]), -gamma2 * pt2[0]], [pt2[1], (A * a1 * pt2[1]) / (1 + a1 * pt2[1])**2]])
            trace_J2 = np.trace(J2)
            det_J2 = np.linalg.det(J2)
            if prev_trace is not None:
                if (prev_trace < 0 and trace_J2 >= 0) or (prev_trace > 0 and trace_J2 <= 0):
                    found_ac.append(a1)
                    st.success(f"**Found critical threshold $a_c$ ≈ {a1:.4f} at fixed point (H*={pt2[0]:.3f}, R*={pt2[1]:.3f})")
        prev_trace = trace_J2
        prev_a1 = a1
        prev_pt = pt2
    if len(found_ac) == 0:
        st.info("No critical thresholds found in the range of a. The system remains stable for all tested values.")
    st.subheader("Bifurcation Diagram & Phase Portrait")
    a_sweep = np.linspace(0.01, 0.32, 1000)
    R_stars, R_mins, R_maxs = run_bifurcation_sweep(a_sweep)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(a_sweep, R_stars, 'k--', label=r'unstable Point $R^*$', alpha=0.8)
    ax[0].plot(a_sweep, R_maxs, 'r-', linewidth=1.5, label=r'stable point $R^*$')
    ax[0].plot(a_sweep, R_mins, 'r-', linewidth=1.5)
    for i, ac in enumerate(found_ac):
        ax[0].axvline(ac, color='blue', linestyle=':', label=fr'$a_{{c{i+1}}} ≈ {ac:.4f}$')
    ax[0].set_xlabel('Bleaching Parameter $a$', fontsize=9)
    ax[0].set_ylabel('Rainfall Intensity $R$', fontsize=9)
    ax[0].set_title('Unified Bifurcation Diagram (Subcritical & Supercritical Regimes)', fontsize=10)
    ax[0].tick_params(axis='both', which='major', labelsize=8)
    ax[0].grid(True, linestyle=':', alpha=0.6)
    ax[0].legend(loc='upper left', fontsize=8)
    starting_points = [[2.3, 3.8], [1.5, 4.8], [3.2, 1.8], [2.8, 3.2]]
    for i, pt in enumerate(starting_points):
        t, H, R = solve_rk4(a_float, H_init=pt[0], R_init=pt[1], H0=H0, gamma2=gamma2, A=A, t_max=600, dt=0.01)
        lbl = 'RK4 Trajectories' if i == 0 else ""
        ax[1].plot(H, R, color='gray', alpha=0.6, label=lbl)
        ax[1].scatter(pt[0], pt[1], color='black', s=20, zorder=5)
    found_curr = fixed_points(H0, A, a_float)
    if len(found_curr) > 0:
        found_curr_arr = np.array(found_curr)
        if stable:
            ax[1].scatter(found_curr_arr[:, 0], found_curr_arr[:, 1], color='green', s=50, zorder=6, label=r'Stable Focus $(H^*, R^*)$')
        else:
            ax[1].scatter(found_curr_arr[:, 0], found_curr_arr[:, 1], color='red', s=50, zorder=6, label=r'Unstable Focus $(H^*, R^*)$')

    ax[1].set_xlabel(r'Column Humidity $H$', fontsize=9)
    ax[1].set_ylabel(r'Rainfall Intensity $R$', fontsize=9)
    ax[1].set_title(fr'Phase Portrait ($a = {a_float:.2f} < a_{{c1}}$)', fontsize=10)
    ax[1].tick_params(axis='both', which='major', labelsize=8)
    ax[1].grid(True, linestyle=':', alpha=0.6)
    ax[1].legend(loc='upper right', fontsize=8)
    st.pyplot(fig)