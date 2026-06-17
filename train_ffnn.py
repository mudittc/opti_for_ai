
import os
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
TARGETS = ["y1_noisy","y3_noisy","control"]

class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,3)
        )
    def forward(self,x):
        return self.net(x)

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
    targ_scaled = scaler_y.fit_transform(feat[:,:3])

    combined = np.hstack([targ_scaled, feat_scaled[:,3:4]])

    X,y = make_windows(combined)

    split = int(len(X)*0.7)

    X_train = X[:split].reshape(split,-1)
    y_train = y[:split]

    model = FFNN()
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

    for epoch in range(EPOCHS):
        for xb,yb in loader:
            pred = model(xb)
            loss = loss_fn(pred,yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    history = combined[:WINDOW].copy()
    preds=[]

    for i in range(WINDOW,len(combined)):
        x = torch.FloatTensor(history[-WINDOW:].reshape(1,-1))
        pred = model(x).detach().numpy()[0]

        wind_next = combined[i,3]

        next_row = np.array([pred[0],pred[1],pred[2],wind_next])

        preds.append(pred)
        history = np.vstack([history,next_row])

    preds = scaler_y.inverse_transform(np.array(preds))
    actual = d[TARGETS].values[WINDOW:]

    print(f"\n{sim}")
    print("RMSE:", np.sqrt(mean_squared_error(actual,preds)))
    print("MAE :", mean_absolute_error(actual,preds))

    fig = go.Figure()

    # True states (physics)
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

    # Actual noisy sensor readings
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

    # FFNN predictions
    fig.add_trace(go.Scatter(
        y=preds[:,0],
        name="Predicted y1 (FFNN)"
    ))

    fig.add_trace(go.Scatter(
        y=preds[:,1],
        name="Predicted y3 (FFNN)"
    ))

    # Control comparison
    fig.add_trace(go.Scatter(
        y=d["control"][WINDOW:],
        name="Actual Control"
    ))

    fig.add_trace(go.Scatter(
        y=preds[:,2],
        name="Predicted Control (FFNN)"
    ))

    fig.update_layout(
        title=f"{sim} - FFNN Recursive Prediction",
        xaxis_title="Timestep",
        yaxis_title="Value",
        hovermode="x unified"
    )

    fig.show(renderer="browser")
