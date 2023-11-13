import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DataFile = "./Data/rpi_data_compact.csv"

with open(DataFile) as f:
    df_compact = pd.read_csv(f)
print(df_compact.head(1))


df_compact.drop(["Unnamed: 0"], axis=1, inplace=True)
print(df_compact)

NaNs_in_df = df_compact.isnull()
print(type(NaNs_in_df))
print(NaNs_in_df.head())

NaNs_per_column = NaNs_in_df.sum()
print(type(NaNs_per_column))
print(NaNs_per_column.head())

NaNs_total = NaNs_per_column.sum()
print(NaNs_total)

NaNs_pct = np.round(
    df_compact.isnull().sum().sum()
    / float(len(df_compact) * len(df_compact.columns))
    * 100,
    decimals=4,
)
print(
    "The DataFrame contains : {} NaNs, equal to {} of the measurements".format(
        NaNs_total, NaNs_pct
    )
)  # EDL : moved parenthesis

df_compact_clean = df_compact.dropna()
print(df_compact_clean)

print(df_compact.dtypes)

df_compact_clean = df_compact_clean.reindex(
    columns=["Date", "Time", "Ping (ms)", "Download (Mbit/s)", "Upload (Mbit/s)"]
)
print(df_compact_clean.head())

df_clean = df_compact_clean

means = df_clean.mean()
stands = df_clean.std()

# Place mean and std for each column in a tuple
stats_ping = (means["Ping (ms)"], stands["Ping (ms)"])
stats_download = (means["Download (Mbit/s)"], stands["Download (Mbit/s)"])
stats_upload = (means["Upload (Mbit/s)"], stands["Upload (Mbit/s)"])

# Print the mean value ± the standard deviation, including measuring units
print("Average ping time: {} ± {} ms".format(stats_ping[0], stats_ping[1]))
print("Average download speed: {} ± {} Mbit/s".format(*stats_download))
print("Average upload speed: {} ± {} Mbit/s".format(*stats_upload))

mins = df_clean.min()
maxs = df_clean.max()

# Place mean and std for each column in a tuple
mima_ping = (mins["Ping (ms)"], maxs["Ping (ms)"])
mima_download = (mins["Download (Mbit/s)"], maxs["Download (Mbit/s)"])
mima_upload = (mins["Upload (Mbit/s)"], maxs["Upload (Mbit/s)"])

# Print the mean and max values, including measuring units
print("Min ping time: {} ms. Max ping time: {} ms".format(*mima_ping))
print(
    "Min download speed: {} Mbit/s. Max download speed: {} Mbit/s".format(
        *mima_download
    )
)
print("Min upload speed: {} Mbit/s. Max upload speed: {} Mbit/s".format(*mima_upload))

print(df_clean.describe())

# Find the min and max ping time
argmin_ping = df_clean["Ping (ms)"].argmin()
argmax_ping = df_clean["Ping (ms)"].argmax()

# Find the min and max download speed
argmin_download = df_clean["Download (Mbit/s)"].argmin()
argmax_download = df_clean["Download (Mbit/s)"].argmax()

# Find the min and max upload speed
argmin_upload = df_clean["Upload (Mbit/s)"].argmin()
argmax_upload = df_clean["Upload (Mbit/s)"].argmax()

df = pd.DataFrame({"field_1": [0, 1], "field_2": [0, 2]})
df.head()

print(df.iloc[1]["field_1"])


print(
    "Ping measure reached minimum on {} at {}".format(
        df_clean.loc[argmin_ping].Date, df_clean.loc[argmin_ping].Time
    )
)

print(
    "Download measure reached minimum on {} at {}".format(
        df_clean.loc[argmin_download].Date, df_clean.loc[argmin_download].Time
    )
)

print(
    "Upload measure reached minimum on {} at {}".format(
        df_clean.loc[argmin_upload].Date, df_clean.loc[argmin_upload].Time
    )
)

print(
    "Ping measure reached maximum on {} at {}".format(
        df_clean.loc[argmax_ping].Date, df_clean.loc[argmax_ping].Time
    )
)

print(
    "Download measure reached maximum on {} at {}".format(
        df_clean.loc[argmax_download].Date, df_clean.loc[argmax_download].Time
    )
)

print(
    "Upload measure reached maximum on {} at {}".format(
        df_clean.loc[argmax_upload].Date, df_clean.loc[argmax_upload].Time
    )
)

df_corr = df_clean.corr()
print(df_corr)

corr = df_corr.values
print("Correlation coefficient between ping and download: {}".format(corr[0, 1]))
print("Correlation coefficient between ping and upload: {}".format(corr[0, 2]))
print("Correlation coefficient between upload and download: {}".format(corr[2, 1]))

fig, ax = plt.subplots(figsize=(10, 5))

# Create x-axis
t = pd.to_datetime(df_clean["Time"])


# Plot three curves of different colors
ax.plot(t, df_clean["Ping (ms)"], label="Ping (ms)")
ax.plot(t, df_clean["Download (Mbit/s)"], label="Download (Mbit/s)")
ax.plot(t, df_clean["Upload (Mbit/s)"], label="Upload (Mbit/s)")

# Insert legend
ax.legend()

ax.set_xlabel("Time (hh:mm:ss)", size=16)
ax.set_ylabel("Measurement", size=16)
ax.set_title("Internet speed demo", size=16)
ax.tick_params(labelsize=14)

plt.show()

with plt.style.context("fivethirtyeight"):
    nbins = 100
    # Initialize figure
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0][0].hist(df_clean["Ping (ms)"], nbins)
    ax[0][0].set_xlabel("Ping (ms)", fontsize=16)
    ax[0][0].tick_params(labelsize=14)
    ax[0][1].hist(df_clean["Upload (Mbit/s)"], nbins)
    ax[0][1].set_xlabel("Upload (Mbit/s)", fontsize=16)
    ax[0][1].tick_params(labelsize=14)
    ax[1][0].hist(df_clean["Download (Mbit/s)"], nbins)
    ax[1][0].set_xlabel("Download (Mbit/s)", fontsize=16)
    ax[1][0].tick_params(labelsize=14)
    ax[1][1].set_visible(False)
plt.show()
