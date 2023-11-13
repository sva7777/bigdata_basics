import datetime
import csv
import pandas as pd
import numpy as np

data_file = "./Data/rpi_data_long.csv"

column_names = [
    "Type A",
    "Measure A",
    "Units A",
    "Type B",
    "Measure B",
    "Units B",
    "Type C",
    "Measure C",
    "Units C",
    "Datetime",
]
with open(data_file, "r") as f:
    df_redundant = pd.read_csv(f, names=column_names)

df_compact = df_redundant.copy()
df_compact.rename(
    columns={
        "Measure A": "Ping (ms)",
        "Measure B": "Download (Mbit/s)",
        "Measure C": "Upload (Mbit/s)",
    },
    inplace=True,
)

df_compact.drop(
    ["Type A", "Type B", "Type C", "Units A", "Units B", "Units C"],
    axis=1,
    inplace=True,
)
df_compact.head()

df_compact["Date"] = df_compact["Datetime"].apply(
    lambda dt_str: pd.to_datetime(dt_str).date()
)
temp = df_compact["Datetime"].apply(lambda dt_str: pd.to_datetime(dt_str))
df_compact["Time"] = temp.dt.time

del df_compact["Datetime"]

df_compact.to_csv("./Data/rpi_data_compact.csv")
