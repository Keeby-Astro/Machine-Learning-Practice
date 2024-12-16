import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Read the data from the text file.
df = pd.read_csv(
    "solar_flux.txt",
    skiprows=2,  # Skip the top two header lines
    header=None, # No header in the data lines
    names=["JulianDate", "Rotation", "Year", "Month", "Day", "Obs", "Adj", "URSI-D"],
    sep=",",
    engine='python'
)

# Convert the numeric columns from strings if necessary (usually pandas does this automatically)
df["JulianDate"] = pd.to_numeric(df["JulianDate"], errors='coerce')
df["Obs"] = pd.to_numeric(df["Obs"], errors='coerce')
df["Adj"] = pd.to_numeric(df["Adj"], errors='coerce')
df["URSI-D"] = pd.to_numeric(df["URSI-D"], errors='coerce')

# Replace zeros with NaN
columns_to_clean = ["Obs", "Adj", "URSI-D"]
df[columns_to_clean] = df[columns_to_clean].replace(0, pd.NA)

# Remove values above 355 and replace with NaN
df[columns_to_clean] = df[columns_to_clean].where((df[columns_to_clean] <= 355) & (df[columns_to_clean].notna()), pd.NA)

# Backward fill the first value if it is NaN or zero
for col in columns_to_clean:
    if pd.isna(df.at[0, col]) or df.at[0, col] == 0:
        df.at[0, col] = df[col].bfill().iloc[0]

# Forward fill the NaN values
df[columns_to_clean] = df[columns_to_clean].ffill().infer_objects(copy=False)

# Ensure no NAType values remain
df[columns_to_clean] = df[columns_to_clean].fillna(0).infer_objects(copy=False)

# Infer objects to ensure no downcasting issues
df = df.infer_objects()

# Create a datetime column from Year, Month, and Day
df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

# Plot the data together
plt.figure(figsize=(10, 6))
plt.plot(df["Date"], df["Obs"], label="Observed Flux Density")
plt.plot(df["Date"], df["Adj"], label="Adjusted Flux Density (1 A.U.)")
plt.plot(df["Date"], df["URSI-D"], label="URSI-D (Adj. x 0.9)")

# Add labels and title
plt.xlabel("Date")
plt.ylabel(f"Flux Density [Solar Flux Unit (10$^{{-22}}$ W⋅m$^{{-2}}$⋅Hz$^{{-1}}$)]")
plt.title("Solar Flux Measurements")

# Add a grid and legend
plt.legend()

# Show the plot
plt.show()

# Plot each data series separately
fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# Observed Flux Density
axs[0].plot(df["Date"], df["Obs"], label="Observed Flux Density", color='b')
axs[0].set_ylabel("Observed Flux Density")
axs[0].legend()

# Adjusted Flux Density (1 A.U.)
axs[1].plot(df["Date"], df["Adj"], label="Adjusted Flux Density (1 A.U.)", color='g')
axs[1].set_ylabel("Adjusted Flux Density (1 A.U.)")
axs[1].legend()

# URSI-D (Adj. x 0.9)
axs[2].plot(df["Date"], df["URSI-D"], label="URSI-D (Adj. x 0.9)", color='r')
axs[2].set_xlabel("Date")
axs[2].set_ylabel("URSI-D (Adj. x 0.9)")
axs[2].legend()

# Add a title to the figure
fig.suptitle("Solar Flux Measurements")

# Show the plots
plt.show()