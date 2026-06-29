import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# -------------------------
# Parameters (tuned for visible motion)
# -------------------------
M1, M2 = 1000, 100
# M1 → mass of the building
# M2 → mass of the damper


k1, k2 = 2000, 1500
# k1 → stiffness between building and ground
# k2 → stiffness between building and damper
# Higher k → stiffer → less displacement | Lower k → more flexible → more movement



b1, b2 = 200, 150
# b1 → damping between building and ground
# b2 → damping between building and damper
# Removes energy from system and reduces oscillations


K = np.array([500, 200, -300, -100])
#   x1   → react strongly to building displacement
#   x2   → react to building velocity
#   x3   → react to damper position
#   x4   → react to damper velocity


# Noise levels (bell gaussian distribution)
sigma_x1 = 0.01
sigma_x3 = 0.01

# -------------------------
# Wind models
# -------------------------
# -------------------------
# Wind models (Ornstein-Uhlenbeck Process)
# -------------------------
def generate_ou_wind(t, mean, theta, sigma, seed=None):
    if seed is not None:
        np.random.seed(seed)
    w = np.zeros(len(t))
    w[0] = mean
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        w[i] = w[i-1] + theta * (mean - w[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return w

# -------------------------
# System dynamics
# -------------------------
def model(t, x, wind_interpolator):

    x1, x2, x3, x4 = x
    w = wind_interpolator(t)

    u = -K @ x

    dx1 = x2
    dx2 = (-(k1+k2)*x1 - (b1+b2)*x2 + k2*x3 + b2*x4 + w - u) / M1
    dx3 = x4
    dx4 = (k2*x1 + b2*x2 - k2*x3 - b2*x4 + u) / M2

    return [dx1, dx2, dx3, dx4]

# -------------------------
# Simulation setup
# -------------------------
t = np.linspace(0, 40, 800)   # 800 points (within your 500–1000 range)
x0 = [0, 0, 0, 0]

# Define randomized wind profiles using Ornstein-Uhlenbeck parameters
wind_profiles = {
    "light_wind":  {"mean": 300, "theta": 0.5, "sigma": 50,  "seed": 42},
    "medium_wind": {"mean": 500, "theta": 0.5, "sigma": 100, "seed": 43},
    "gusty_wind":  {"mean": 400, "theta": 0.1, "sigma": 250, "seed": 44},
    "storm":       {"mean": 700, "theta": 0.8, "sigma": 400, "seed": 45}
}

all_data = []

# -------------------------
# Run simulations
# -------------------------
for w_type, params in wind_profiles.items():

    # Pre-generate randomized wind time-series
    wind_vals = generate_ou_wind(
        t, 
        mean=params["mean"], 
        theta=params["theta"], 
        sigma=params["sigma"], 
        seed=params["seed"]
    )
    
    # Create continuous interpolator for the ODE solver
    wind_interpolator = lambda ti: np.interp(ti, t, wind_vals)

    # Solve dynamics
    sol = solve_ivp(model, [0, 40], x0, t_eval=t, args=(wind_interpolator,))
    X = sol.y.T   # (N, 4)

    # Control input values
    control_vals = np.array([-K @ X[i] for i in range(len(t))])

    # Noise (separate distributions)
    noise_x1 = np.random.normal(0, sigma_x1, len(t))
    noise_x3 = np.random.normal(0, sigma_x3, len(t))

    y1 = X[:, 0] + noise_x1
    y3 = X[:, 2] + noise_x3

    # -------------------------
    # Create dataset
    # -------------------------
    df = pd.DataFrame({
        "time": t,
        "x1": X[:, 0],
        "x2": X[:, 1],
        "x3": X[:, 2],
        "x4": X[:, 3],
        "y1_noisy": y1,
        "y3_noisy": y3,
        "wind": wind_vals,
        "control": control_vals,
        "simulation_type": w_type
    })

    all_data.append(df)

    # -------------------------
    # Plot (one graph per wind)
    # -------------------------
    fig = go.Figure()

    # STATES (left axis)
    fig.add_trace(go.Scatter(x=t, y=X[:,0], name="x1 (building)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=t, y=X[:,1], name="x2 (velocity)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=t, y=X[:,2], name="x3 (damper)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=t, y=X[:,3], name="x4 (damper vel)", line=dict(dash="dot")))

    # NOISY OUTPUTS
    fig.add_trace(go.Scatter(x=t, y=y1, name="x1 noisy", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=y3, name="x3 noisy", line=dict(dash="dash")))

    # INPUTS (right axis)
    fig.add_trace(go.Scatter(
        x=t, y=wind_vals,
        name="wind",
        yaxis="y2",
        line=dict(dash="longdash")
    ))

    fig.add_trace(go.Scatter(
        x=t, y=control_vals,
        name="control (active damping)",
        yaxis="y2"
    ))

    # LAYOUT
    fig.update_layout(
        title=f"System Response - {w_type}",
        xaxis=dict(title="Time"),

        yaxis=dict(
            title="States / Displacement",
            side="left"
        ),

        yaxis2=dict(
            title="Input Forces (Wind & Control)",
            overlaying="y",
            side="right"
        ),

        hovermode="x unified"
    )

    # Show in browser
    fig.show(renderer="browser")

    # Optional save
    # fig.write_html(f"plot_{w_type}.html")

# -------------------------
# Combine all data
# -------------------------
combined_df = pd.concat(all_data, ignore_index=True)

# Save single CSV
combined_df.to_csv("final_dataset.csv", index=False)

print("Dataset saved as final_dataset.csv")