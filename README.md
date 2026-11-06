# 🔬 Laser & Atmospheric Dynamical Systems Analysis

An interactive, high-performance web application built with **Streamlit** to analyze bifurcation phenomena and phase space dynamics in laser physics and atmospheric moisture models. 

👉 **Live App:** [View on Streamlit Community Cloud](https://your-app-url.streamlit.app) *(Replace with your actual link after deploying!)*

---

## 🌟 Key Features
- **Interactive Controls:** Real-time adjustment of pump rates ($N_0$), gain coefficients ($\gamma$), aerosol suppression ($A$), and cloud invigoration sensitivity ($a$).
- **Blazing-Fast Numerical Engine:** Utilizes **Numba (`@njit`)** compilation to run heavy fixed-step 4th-order Runge-Kutta (RK4) ODE solvers instantly during parameter sweeps.
- **Side-by-Side Visualization:** Generates publication-quality, side-by-side Matplotlib subplots (`ax[0]` and `ax[1]`) featuring:
  - **Unified Bifurcation Diagrams** mapping stable and unstable fixed-point branches alongside critical thresholds ($a_c$).
  - **Dynamic Phase Portraits** tracing multi-trajectory convergence toward stable focuses.
- **Theme-Aware Styling:** Transparent figure backdrops with dynamic text, ticks, and spine coloring optimized for both light and dark modes.

---

## ⚙️ Mathematical Model

The system is governed by coupled 2D nonlinear rate equations modeling feedback loops in atmospheric moisture/laser gain dynamics:

$$f(N, I) = \gamma (N_0 - N - NI)$$
$$g(N, I) = I \left( N - 1 - \frac{A}{1 + aI} \right)$$

---

## 🚀 Local Installation & Execution

If you want to run this application locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Laser_atmosphere_model.git](https://github.com/your-username/Laser_atmosphere_model.git)
   cd Laser_atmosphere_model
