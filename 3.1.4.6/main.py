import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./Data/rpi_describe.csv")

df['rounded_ping'] = df['Ping (ms)'].round(2)
df['diff_upload_minus_download'] = df['Upload (Mbit/s)'] - df['Download (Mbit/s)']

count = df['rounded_ping'].count()
mean =  df['rounded_ping'].mean()
median = df['rounded_ping'].median()
std = df['rounded_ping'].std()
rng = df['rounded_ping'].max() -df['rounded_ping'].min()


countstring = 'The count of the distribution is {}'.format(count)
meanstring =  'The mean of the distribution is {:.2f}'.format(mean)
stdstring = 'The standart deviation of the  distribution is {:.2f}'.format(std)
rangestring = 'The range between {:.2f} and {:.2f} is {:.2f}'.format(df['rounded_ping'].max(),df['rounded_ping'].min(), rng)

print(countstring)
print(meanstring)
print(stdstring)
print(rangestring)

freq = df['rounded_ping'].value_counts()
print(freq)
print(type(freq))
freq = freq.to_frame().reset_index()

freq.columns = ['value', 'freq']
print(freq)

plt.figure(figsize=(20,10))
plt.ylabel('Frequency')
plt.xlabel('Weight')
plt.plot(freq.value,freq.freq, "o", markersize = 10, color = 'g')

plt.show()
