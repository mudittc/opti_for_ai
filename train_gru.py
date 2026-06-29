
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

CSV_FILE = "final_dataset.csv"
WINDOW = 3
EPOCHS = 100
BATCH_SIZE = 32

FEATURES = ["y1_noisy","y3_noisy","control","wind"]
TARGETS = ["x1","x3","control"]

class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=4,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(64,3)

    def forward(self,x):
        out,_ = self.gru(x)
        return self.fc(out[:,-1,:])

def make_windows(features, targets):

    X = []
    y = []

    for i in range(len(features) - WINDOW):

        X.append(features[i:i+WINDOW])
        y.append(targets[i+WINDOW])

    return np.array(X), np.array(y)

df = pd.read_csv(CSV_FILE)

# -------------------------
# Prepare data
# -------------------------
feat = df[FEATURES].values

scaler_x = StandardScaler()
scaler_y = StandardScaler()

feat_scaled = scaler_x.fit_transform(feat)

targets = df[TARGETS].values
targ_scaled = scaler_y.fit_transform(targets)


X, y = make_windows(feat_scaled, targ_scaled)

# -------------------------
# Train / Validation / Test split
# -------------------------
train_split = int(len(X) * 0.70)
val_split = int(len(X) * 0.85)

X_train = X[:train_split]
y_train = y[:train_split]

X_val = X[train_split:val_split]
y_val = y[train_split:val_split]

X_test = X[val_split:]
y_test = y[val_split:]

# -------------------------
# Build model
# -------------------------
model = GRUNet()

opt = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

loss_fn = nn.MSELoss()

loader = DataLoader(
    TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

best_val_loss = float("inf")
patience = 20
counter = 0

train_losses = []
val_losses = []

# -------------------------
# Training
# -------------------------
for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for xb, yb in loader:

        pred = model(xb)

        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

    train_loss = epoch_loss / len(loader)
    train_losses.append(train_loss)

    model.eval()

    with torch.no_grad():

        val_pred = model(torch.FloatTensor(X_val))

        val_loss = loss_fn(
            val_pred,
            torch.FloatTensor(y_val)
        ).item()

    val_losses.append(val_loss)

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0

    else:

        counter += 1

    if counter >= patience:

        print("Early Stopping Triggered")
        break

test_indices = np.arange(val_split + WINDOW, len(df))

y1_test = df.loc[test_indices, "y1_noisy"].values
y3_test = df.loc[test_indices, "y3_noisy"].values

# -------------------------
# Loss Plot
# -------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Training")

plt.legend()
plt.grid()

plt.show()

model.eval()

with torch.no_grad():

    preds = model(
        torch.FloatTensor(X_test)
    ).numpy()

preds = scaler_y.inverse_transform(preds)
actual = scaler_y.inverse_transform(y_test)

# -------------------------
# Metrics
# -------------------------
print("\nModel Performance")

names = ["x1", "x3", "control"]

for i, name in enumerate(names):

    rmse = np.sqrt(
        mean_squared_error(
            actual[:,i],
            preds[:,i]
        )
    )

    mae = mean_absolute_error(
        actual[:,i],
        preds[:,i]
    )

    print("----------------")

    print(name)

    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")

# -------------------------
# Prediction Plot
# -------------------------
fig = go.Figure()

# True States
fig.add_trace(go.Scatter(
    y=actual[:,0],
    name="True x1 (Building)",
    line=dict(width=3)
))

fig.add_trace(go.Scatter(
    y=actual[:,1],
    name="True x3 (Damper)",
    line=dict(width=3)
))

fig.add_trace(go.Scatter(
    y=y1_test,
    name="Measured y1",
    line=dict(dash="dot")
))

fig.add_trace(go.Scatter(
    y=y3_test,
    name="Measured y3",
    line=dict(dash="dot")
))

# Predicted States
fig.add_trace(go.Scatter(
    y=preds[:,0],
    name="Predicted x1",
    line=dict(dash="dash")
))

fig.add_trace(go.Scatter(
    y=preds[:,1],
    name="Predicted x3",
    line=dict(dash="dash")
))

# Control
fig.add_trace(go.Scatter(
    y=actual[:,2],
    name="True Control",
    line=dict(width=3)
))

fig.add_trace(go.Scatter(
    y=preds[:,2],
    name="Predicted Control",
    line=dict(dash="dash")
))

fig.update_layout(

    title="GRU One-Step Ahead Forecast",

    xaxis_title="Test Sample",

    yaxis_title="Value",

    hovermode="x unified",

    template="plotly_white"

)

fig.show(renderer="browser")