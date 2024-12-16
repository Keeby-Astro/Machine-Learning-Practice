# Time Series Forecasting of 10.7-cm Solar Radio Flux with CNN-Transformer Model

'''
Purpose: The purpose of this script is to forecast the 10.7-cm Solar Radio Flux using a hybrid CNN-Transformer model.
         This script will preprocess the data, create a CNN-Transformer model, and train the model to forecast
         the 10.7-cm Solar Radio Flux. The script will also evaluate the model's performance and visualize the results
         for each column in the dataset (Observed Flux Density, Adjusted Flux Density, and URSI-D (Adjusted x 0.9)).

Inputs: solar_flux.txt

Outputs: Residual Plot, Prediction vs Actual Scatter Plot, Error Histogram, Loss Curves, and Predictions for each column

Notes: The dataset contains the following columns:
         - JulianDate: Julian Date of the observation
         - Rotation: Solar rotation number
         - Year: Year of the observation
         - Month: Month of the observation
         - Day: Day of the observation
         - Obs: Observed Flux Density
         - Adj: Adjusted Flux Density (1 A.U.)
         - URSI-D: URSI-D (Adjusted x 0.9)
'''

# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess the solar flux data
df = pd.read_csv(
    "solar_flux.txt",
    skiprows=2,
    header=None,
    names=["JulianDate", "Rotation", "Year", "Month", "Day", "Obs", "Adj", "URSI-D"],
    sep=",",
    engine='python'
)

# Convert numeric columns
df["JulianDate"] = pd.to_numeric(df["JulianDate"], errors='coerce')
df[["Obs", "Adj", "URSI-D"]] = df[["Obs", "Adj", "URSI-D"]].apply(pd.to_numeric, errors='coerce')

# Replace zeros and fill missing values
columns_to_clean = ["Obs", "Adj", "URSI-D"]
df[columns_to_clean] = df[columns_to_clean].replace(0, pd.NA)
for col in columns_to_clean:
    if pd.isna(df.at[0, col]) or df.at[0, col] == 0:
        df.at[0, col] = df[col].bfill().iloc[0]
df[columns_to_clean] = df[columns_to_clean].ffill().bfill().fillna(0).infer_objects(copy=False)

# Create a datetime column
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
df = df.dropna(subset=["Date"])
df.set_index("Date", inplace=True)

# Resample to daily averages
df = df.resample("1D").mean()
df = df.dropna(how='all', subset=columns_to_clean)

# CNN-Transformer Hybrid Model for Time Series Forecasting
'''
CNN-Transformer Model:
Purpose: The CNN-Transformer model is designed to capture both local and global patterns in time series data.
         The model consists of a CNN encoder followed by a Transformer encoder to learn the temporal dependencies
         and make predictions for the next time step.
'''
class CNNTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        CNN Model:
        - Conv1D Layer with 16 filters, kernel size of 3, and padding of 1
        - ReLU Activation
        - Conv1D Layer with 32 filters, kernel size of 3, and padding of 1
        - ReLU Activation
        - Conv1D Layer with 64 filters, kernel size of 3, and padding of 1
        - ReLU Activation
        - Conv1D Layer with 128 filters, kernel size of 3, and padding of 1
        '''
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        '''
        Transformer Model:
        - Linear Layer to embed the input
        - Transformer Encoder with 2 layers, 256 hidden units, 4 attention heads, and 0.1 dropout
        - Linear Layer for final prediction
        '''
        self.embedding = nn.Linear(128, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(256, 1)

    '''
    Forward Pass of the Model:
    Args:
        Inputs:
        x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
        Output tensor of shape (batch_size, seq_len, 1)
    '''
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.transpose(1, 2)
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# Define columns to predict
columns_to_predict = ["Adj", "Obs", "URSI-D"]
results = {}

# Loop through each column
for column in columns_to_predict:
    print(f"Processing column: {column}")

    # Prepare time series
    timeseries = df[column].values.astype('float32').reshape(-1, 1)
    timeseries[0] = timeseries[1]
    tensor_timeseries = torch.tensor(timeseries, dtype=torch.float32).to(device)
    min_val, max_val = torch.min(tensor_timeseries), torch.max(tensor_timeseries)
    norm_timeseries = (tensor_timeseries - min_val) / (max_val - min_val)

    # Train, validation, and test split
    train_size = int(len(norm_timeseries) * 0.65)
    val_size = int(len(norm_timeseries) * 0.15)
    test_size = len(norm_timeseries) - train_size - val_size

    # Split the data
    train = norm_timeseries[:train_size]
    val = norm_timeseries[train_size:train_size + val_size]
    test = norm_timeseries[train_size + val_size:]

    # Create DataLoader for training and validation
    X_train, y_train = train[:-1].unsqueeze(-1), train[1:]
    X_val, y_val = val[:-1].unsqueeze(-1), val[1:]
    X_test, y_test = test[:-1].unsqueeze(-1), test[1:]

    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=512)
    val_loader = data.DataLoader(data.TensorDataset(X_val, y_val), shuffle=False, batch_size=512)

    # Initialize the model, optimizer, loss function, and number of epochs
    model = CNNTransformerModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    n_epochs = 25

    # Initialize lists to store losses
    train_losses, val_losses, test_losses = [], [], []

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = torch.sqrt(criterion(model(X_train), y_train))
            val_loss = torch.sqrt(criterion(model(X_val), y_val))
            test_loss = torch.sqrt(criterion(model(X_test), y_test))
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        test_losses.append(test_loss.item())

        # Print losses every 5 epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Store the results
    with torch.no_grad():
        train_pred = model(X_train)
        val_pred = model(X_val)
        test_pred = model(X_test)

    # Inverse transform the predictions and targets
    train_pred = (train_pred * (max_val - min_val) + min_val).cpu().numpy()
    val_pred = (val_pred * (max_val - min_val) + min_val).cpu().numpy()
    test_pred = (test_pred * (max_val - min_val) + min_val).cpu().numpy()
    y_train = (y_train * (max_val - min_val) + min_val).cpu().numpy()
    y_val = (y_val * (max_val - min_val) + min_val).cpu().numpy()
    y_test = (y_test * (max_val - min_val) + min_val).cpu().numpy()

    # Residuals
    train_residuals = y_train.flatten() - train_pred.flatten()
    val_residuals = y_val.flatten() - val_pred.flatten()
    test_residuals = y_test.flatten() - test_pred.flatten()

    # Residual Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(train_residuals)), train_residuals, alpha=0.5, label="Train Residuals", color="red")
    plt.scatter(range(len(val_residuals)), val_residuals, alpha=0.5, label="Validation Residuals", color="orange")
    plt.scatter(range(len(test_residuals)), test_residuals, alpha=0.5, label="Test Residuals", color="green")
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.legend()
    plt.title(f"Residual Plot for {column}")
    plt.xlabel("Samples")
    plt.ylabel("Residuals")
    plt.show()

    # Prediction vs Actual Scatter Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(y_train, train_pred, alpha=0.5, label="Train", color="red")
    plt.scatter(y_val, val_pred, alpha=0.5, label="Validation", color="orange")
    plt.scatter(y_test, test_pred, alpha=0.5, label="Test", color="green")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
    plt.legend()
    plt.title(f"Prediction vs Actual Values for {column}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

    # Error Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(train_residuals, bins=30, alpha=0.5, label="Train Residuals", color="red")
    plt.hist(val_residuals, bins=30, alpha=0.5, label="Validation Residuals", color="orange")
    plt.hist(test_residuals, bins=30, alpha=0.5, label="Test Residuals", color="green")
    plt.legend()
    plt.title(f"Error Histogram for {column}")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

    # Loss Curves
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss", color="red")
    plt.plot(val_losses, label="Validation Loss", color="orange")
    plt.plot(test_losses, label="Test Loss", color="green")
    plt.legend()
    plt.title(f"Loss Curve for {column}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    with torch.no_grad():
        train_pred = model(X_train)
        val_pred = model(X_val)
        test_pred = model(X_test)

    train_pred = (train_pred * (max_val - min_val) + min_val).cpu().numpy()
    val_pred = (val_pred * (max_val - min_val) + min_val).cpu().numpy()
    test_pred = (test_pred * (max_val - min_val) + min_val).cpu().numpy()

    # Convert y_train, y_val, y_test to PyTorch tensors for scaling
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Inverse transform the targets
    y_train = (y_train_tensor * (max_val - min_val) + min_val).cpu().numpy()
    y_val = (y_val_tensor * (max_val - min_val) + min_val).cpu().numpy()
    y_test = (y_test_tensor * (max_val - min_val) + min_val).cpu().numpy()

    # Align predictions with the original time series
    train_plot = np.full((len(timeseries), 1), np.nan)
    val_plot = np.full((len(timeseries), 1), np.nan)
    test_plot = np.full((len(timeseries), 1), np.nan)

    train_plot[:len(train_pred), 0] = train_pred.flatten()
    val_plot[train_size:train_size + len(val_pred), 0] = val_pred.flatten()
    test_plot[train_size + val_size:train_size + val_size + len(test_pred), 0] = test_pred.flatten()
    original_plot = np.full((len(timeseries), 1), np.nan)
    original_plot[:train_size + val_size, 0] = timeseries[:train_size + val_size].flatten()

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[:train_size + val_size], original_plot[:train_size + val_size].flatten(),
              label=f"{column} Data", color='blue')
    plt.plot(df.index[:len(train_pred)], train_plot[:len(train_pred)].flatten(),
              label="Train Predictions", color='red')
    plt.plot(df.index[train_size:train_size + len(val_pred)], val_plot[train_size:train_size + len(val_pred)].flatten(),
              label="Validation Predictions", color='orange')
    plt.plot(df.index[train_size + val_size:train_size + val_size + len(test_pred)], test_plot[train_size + val_size:train_size + val_size + len(test_pred)].flatten(),
              label="Test Predictions", color='green')
    plt.legend()
    plt.title(f"Predictions for {column} with CNN-Transformer")
    plt.xlabel("Date")
    plt.ylabel(f"{column} Solar Flux Unit (10$^{{-22}}$ W⋅m$^{{-2}}$⋅Hz$^{{-1}}$)")
    plt.show()

    print(f"Finished processing column: {column}\n")