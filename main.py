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
def wind_model(t, mode):

    if mode == "sin_300":
        return 300 + 200*np.sin(0.4*t)

    elif mode == "sin_500":
        return 500 + 200*np.sin(0.4*t)

    elif mode == "step_slow":
        return 300 if int(t/5) % 2 == 0 else 600

    elif mode == "step_fast":
        return 200 if int(t/2) % 2 == 0 else 700

# -------------------------
# System dynamics
# -------------------------
def model(t, x, wind_type):

    x1, x2, x3, x4 = x
    w = wind_model(t, wind_type)

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

wind_types = ["sin_300", "sin_500", "step_slow", "step_fast"]

all_data = []

# -------------------------
# Run simulations
# -------------------------
for w_type in wind_types:

    sol = solve_ivp(model, [0, 40], x0, t_eval=t, args=(w_type,))
    X = sol.y.T   # (N, 4)

    # Inputs
    wind_vals = np.array([wind_model(ti, w_type) for ti in t])
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