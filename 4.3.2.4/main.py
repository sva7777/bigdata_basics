import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact


def scatter_view(x, y, z, azim, elev):
    # Init figure and axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Compute scatter plot
    ax.scatter(x, y, z)
    ax.set_xlabel("D rate (Mbit/s)", fontsize=16)
    ax.set_ylabel("U rate (Mbit/s)", fontsize=16)
    ax.set_zlabel("P rate (1/s)", fontsize=16)

    # Specify azimuth
    # and elevation
    ax.azim = azim
    ax.elev = elev


# Load internet speed data
df = pd.read_csv("./Data/rpi_data_processed.csv")

# Initialize dataframe df_rates
df_rates = df.drop(["Ping (ms)", "Date", "Time"], axis=1)

# Rename the download and
# upload columns of df_rates
lookup = {"Download (Mbit/s)": "download_rate", "Upload (Mbit/s)": "upload_rate"}
df_rates = df_rates.rename(columns=lookup)

# Calculate ping_rate
ping_rate = 1.0 / df["Ping (ms)"]

# Convert ping_rate to 1/seconds
ping_rate = 1000.0 * ping_rate

# Add a column to complete the task
df_rates["ping_rate"] = ping_rate

print(df_rates.head)


xi = df_rates["download_rate"]
yi = df_rates["upload_rate"]
zi = df_rates["ping_rate"]
interact(
    lambda azim, elev: scatter_view(xi, yi, zi, azim, elev), azim=(0, 90), elev=(0, 90)
)
plt.show()
