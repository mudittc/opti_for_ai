
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

def make_windows(data):
    X,y=[],[]
    for i in range(len(data)-WINDOW):
        X.append(data[i:i+WINDOW])
        y.append(data[i+WINDOW,:3])
    return np.array(X),np.array(y)

df = pd.read_csv(CSV_FILE)

for sim in df["simulation_type"].unique():
    d = df[df["simulation_type"]==sim].reset_index(drop=True)

    feat = d[FEATURES].values

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    feat_scaled = scaler_x.fit_transform(feat)
    targets = d[TARGETS].values
    targ_scaled = scaler_y.fit_transform(targets)

    combined = np.hstack([targ_scaled, feat_scaled[:,3:4]])

    X,y = make_windows(combined)

    train_split = int(len(X)*0.70)
    val_split = int(len(X)*0.85)

    X_train = X[:train_split]
    y_train = y[:train_split]

    X_val = X[train_split:val_split]
    y_val = y[train_split:val_split]

    X_test = X[val_split:]
    y_test = y[val_split:]

    model = GRUNet()

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb,yb in loader:
            pred = model(xb)
            loss = loss_fn(pred,yb)
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

        print(
            f"Epoch {epoch+1} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print("Early Stopping Triggered")
            break


    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(train_losses,label="Training Loss")
    plt.plot(val_losses,label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{sim} Training")
    plt.legend()
    plt.grid()
    plt.show()

    history = combined[:WINDOW].copy()
    preds=[]

    for i in range(WINDOW,len(combined)):
        x = torch.FloatTensor(history[-WINDOW:]).unsqueeze(0)
        pred = model(x).detach().numpy()[0]

        wind_next = combined[i,3]

        next_row = np.array([pred[0],pred[1],pred[2],wind_next])

        preds.append(pred)
        history = np.vstack([history,next_row])

    preds = scaler_y.inverse_transform(np.array(preds))
    actual = d[TARGETS].values[WINDOW:]

    print(f"\n{sim}")
    names = ["x1","x3","control"]

    for i,name in enumerate(names):
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

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=d["x1"][WINDOW:],
        name="True x1 (Building)",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        y=d["x3"][WINDOW:],
        name="True x3 (Damper)",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        y=d["y1_noisy"][WINDOW:],
        name="Actual y1 Noisy",
        line=dict(dash="dot")
    ))

    fig.add_trace(go.Scatter(
        y=d["y3_noisy"][WINDOW:],
        name="Actual y3 Noisy",
        line=dict(dash="dot")
    ))

    fig.add_trace(go.Scatter(
        y=preds[:,0],
        name="Predicted y1 (GRU)"
    ))

    fig.add_trace(go.Scatter(
        y=preds[:,1],
        name="Predicted y3 (GRU)"
    ))

    fig.add_trace(go.Scatter(
        y=d["control"][WINDOW:],
        name="Actual Control"
    ))

    fig.add_trace(go.Scatter(
        y=preds[:,2],
        name="Predicted Control (GRU)"
    ))

    fig.update_layout(
        title=f"{sim} - GRU Recursive Prediction",
        xaxis_title="Timestep",
        yaxis_title="Value",
        hovermode="x unified"
    )

    fig.show(renderer="browser")