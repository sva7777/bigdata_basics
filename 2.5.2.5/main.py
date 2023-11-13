import sqlite3
import pandas as pd
from matplotlib import pyplot as plt

conn = sqlite3.connect("./Data/InternetSpeed.db")
cur = conn.cursor()
cur.execute("SELECT * FROM LA_wifi_speed_UK LIMIT 1")

columns = [member[0] for member in cur.description]
columns = columns[1:]

columns = [c.replace("_p", "") for c in columns]
columns = [c.replace("_d", "") for c in columns]
columns = [c.replace("_u", "") for c in columns]
columns = list(set(columns))

suffix = {"_p": "ping", "_d": "download", "_u": "upload"}

area = columns[0]
print(area)

plt.figure(figsize=(10, 8))

# Plot each variable in suffix.keys() for each area
for s in suffix.keys():
    query = 'select "{}{}" from LA_wifi_speed_UK order by DateTime'.format(area, s)
    cur.execute(query)
    plt.plot(cur.fetchall(), label=suffix[s])
plt.legend()
plt.title(area)
plt.show()

new_columns = ["Area", "Average_p", "Average_d", "Average_u"]

df = pd.DataFrame(columns=new_columns)

print(columns)

for i in range(len(columns) - 1):  # EDL : replace xrange with range

    # It seems there is an error in instructions. Fix this
    if columns[i] == "DateTime":
        continue

    tmp_list = []
    tmp_list.append(columns[i])
    for s in suffix.keys():
        query = "select AVG({}{}) from LA_wifi_speed_UK".format(columns[i], s)
        cur.execute(query)
        mean = cur.fetchone()
        tmp_list.append(mean[0])
    # append the columns to the empty DataFrame
    df = df.append(pd.Series(tmp_list, index=new_columns), ignore_index=True)

# visualize the head of the dataframe here
plt.figure(figsize=(20, 10))
plt.plot(df.index, df[["Average_d", "Average_u", "Average_p"]], "o")
plt.legend(["Average Download", "Average Upload", "Average ping"])
plt.show()

try:
    cur.execute("DROP TABLE average_speed")
except:
    pass

df.to_sql("average_speed", conn)

query = 'SELECT * FROM average_speed JOIN LA_population ON LA_population."LA_code"=average_speed.Area'

cur.execute(query)
k = 0
for row in cur:
    if k > 10:
        break
    print("{}={}".format(k, row))

    k += 1
