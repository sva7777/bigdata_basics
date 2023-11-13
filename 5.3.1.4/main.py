import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

# import folium_utils
plt.style.use("fivethirtyeight")

conn = sqlite3.connect("./Data/InternetSpeed.db")
cur = conn.cursor()

query = "SELECT * FROM average_speed"
df = pd.read_sql(query, conn)

df.drop("index", inplace=True, axis=1)
print(df.shape)
print(df.head())

dfp = df[["Area", "Average_p"]]
dfp = dfp.rename(columns={"Area": "LA_code"})
print(dfp.head())

print(dfp.Average_p.min())
print(dfp.Average_p.max())

p_bins = np.arange(
    dfp.Average_p.min(),
    dfp.Average_p.max(),
    (dfp.Average_p.max() - dfp.Average_p.min()) / 10,
)

p_bins = list(p_bins)
print(p_bins)

import folium
import ast


#  mymap = folium_utils.folium_top_x_preds_mapper(...)


def selected_json_dict_generator(full_list_df, geo_label, geo_label_list):
    allowed_df = full_list_df[full_list_df[geo_label].isin(geo_label_list)]

    output_dict = {
        "crs": {
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            "type": "name",
        },
        "type": "FeatureCollection",
        "features": [
            {
                "geometry": ast.literal_eval(allowed_df["geometry"].iloc[i]),
                "type": "Feature",
                "properties": {geo_label: allowed_df[geo_label].iloc[i]},
                "type": "Feature",
            }
            for i in range(len(allowed_df))
        ],
    }

    return output_dict


# file LA_poligones.json is not correct ;(
la_json = pd.read_json("./Data/LA_poligons.json")

print(la_json.head())

bins = list(dfp.Average_p.quantile([0, 0.1, 0.4, 0.6, 0.8, 1]))
top_x_jsons = selected_json_dict_generator(la_json, "LA_code", dfp["LA_code"].values)
top_x_data = dfp.copy()

m = folium.Map(location=[52.061, -1.336], zoom_start=6)

folium.Choropleth(
    geo_data=top_x_jsons,
    data=top_x_data,
    columns=["LA_code", "Average_p"],
    key_on="feature.id",
    fill_color="BuPu",
    fill_opacity=0.7,
    line_opacity=0.5,
    bins=bins,
).add_to(m)
