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
sigma_x1 = 0.03
sigma_x3 = 0.03

# -------------------------
# Synthetic Wind Generator
# -------------------------
def generate_wind(
    t,
    min_force=200,
    max_force=700,
    alpha=0.995,
    target_std=25,
    target_interval=(150,300),
    turbulence_std=2,
    gust_probability=0.015,
    gust_duration=(20,60),
    gust_strength=(40,120),
    seed=None
):

    if seed is not None:
        np.random.seed(seed)

    n = len(t)

    wind = np.zeros(n)

    target = np.random.uniform(350,550)
    wind[0] = target

    next_target = np.random.randint(*target_interval)

    gust = np.zeros(n)

    # -----------------------------
    # Generate smooth gusts
    # -----------------------------
    i = 0
    while i < n:

        if np.random.rand() < gust_probability:

            duration = np.random.randint(*gust_duration)

            amplitude = np.random.uniform(*gust_strength)

            if np.random.rand() < 0.5:
                amplitude *= -1

            end = min(i + duration, n)

            # Half cosine pulse
            pulse = amplitude * np.sin(
                np.linspace(0, np.pi, end - i)
            )

            gust[i:end] += pulse

            i = end

        else:
            i += 1

    # -----------------------------
    # Wind evolution
    # -----------------------------
    for i in range(1, n):

        if i >= next_target:

            target += np.random.normal(0, target_std)
            target = np.clip(target, min_force, max_force)

            next_target += np.random.randint(*target_interval)

        turbulence = np.random.normal(0, turbulence_std)

        wind[i] = (
            alpha * wind[i-1]
            + (1-alpha) * target
            + gust[i]
            + turbulence
        )

        wind[i] = np.clip(
            wind[i],
            min_force,
            max_force
        )

    return wind

# -------------------------
# System dynamics
# -------------------------
def model(t, x, t_grid, wind_profile):

    x1, x2, x3, x4 = x
    w = np.interp(t, t_grid, wind_profile)

    u = -K @ x

    dx1 = x2
    dx2 = (-(k1+k2)*x1 - (b1+b2)*x2 + k2*x3 + b2*x4 + w - u) / M1
    dx3 = x4
    dx4 = (k2*x1 + b2*x2 - k2*x3 - b2*x4 + u) / M2

    return [dx1, dx2, dx3, dx4]

# -------------------------
# Simulation setup
# -------------------------
t = np.linspace(0, 40, 1500)   # 800 points (within your 500–1000 range)
x0 = [0, 0, 0, 0]

# -------------------------
# Generate fixed synthetic wind profile
# -------------------------
wind_vals = generate_wind(t, seed=None)

# -------------------------
# Solve system
# -------------------------
sol = solve_ivp(
    model,
    [0, 40],
    x0,
    t_eval=t,
    args=(t, wind_vals)
)

X = sol.y.T

# -------------------------
# Active control force
# -------------------------
control_vals = np.array([-K @ X[i] for i in range(len(t))])

# -------------------------
# Sensor noise
# -------------------------
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
    "control": control_vals
})

# -------------------------
# Plot
# -------------------------
fig = go.Figure()

# True states
fig.add_trace(go.Scatter(
    x=t,
    y=X[:, 0],
    name="x1 (building)",
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=t,
    y=X[:, 1],
    name="x2 (velocity)",
    line=dict(dash="dot")
))

fig.add_trace(go.Scatter(
    x=t,
    y=X[:, 2],
    name="x3 (damper)",
    line=dict(width=2)
))

fig.add_trace(go.Scatter(
    x=t,
    y=X[:, 3],
    name="x4 (damper velocity)",
    line=dict(dash="dot")
))

# Noisy measurements
fig.add_trace(go.Scatter(
    x=t,
    y=y1,
    name="x1 noisy",
    line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    x=t,
    y=y3,
    name="x3 noisy",
    line=dict(dash="dash")
))

# Wind
fig.add_trace(go.Scatter(
    x=t,
    y=wind_vals,
    name="Wind Force",
    yaxis="y2",
    line=dict(dash="longdash")
))

# Control
fig.add_trace(go.Scatter(
    x=t,
    y=control_vals,
    name="Control Force",
    yaxis="y2"
))

fig.update_layout(

    title="System Response under Synthetic Wind",

    xaxis=dict(
        title="Time (s)"
    ),

    yaxis=dict(
        title="System States",
        side="left"
    ),

    yaxis2=dict(
        title="Wind / Control Force (N)",
        overlaying="y",
        side="right"
    ),

    hovermode="x unified"
)

fig.show(renderer="browser")

# -------------------------
# Save dataset
# -------------------------
df.to_csv("final_dataset.csv", index=False)

print("Dataset saved successfully as final_dataset.csv.")