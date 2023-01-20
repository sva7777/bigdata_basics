# import ...
import pandas as pd
from matplotlib import pylab as plt
import numpy as np

# Choose a style sheet to use from the matplotlib or use the method shown in the chapter to create a custom style sheet.
plt.style.use('fivethirtyeight')

# Import data from csv file into a dataframe and display the first few rows
df_compact = pd.read_csv('./Data/rpi_data_processed.csv')
print( df_compact.info())

print( df_compact.isnull().sum().sum() )

# Initialize figure
fig, ax = plt.subplots(figsize=(10, 5))

# Plot three curves of different colors
t = pd.to_datetime(df_compact['Time'])
ax.plot(t, df_compact['Ping (ms)'], label='Ping (ms)')
ax.plot(t, df_compact['Upload (Mbit/s)'], label='Upload (Mbit/s)')
ax.plot(t, df_compact['Download (Mbit/s)'], label='Download (Mbit/s)')

# Insert a legend outside of the main plot
ax.legend(bbox_to_anchor=(1.3, 1.))
plt.show()

acceptable_upload = 40 #Mbit/s
acceptable_download = 30 #Mbit/s
acceptable_ping = 20 #ms


# Initialize figure
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# plot ping speed
ax[0][0].hist(df_compact['Ping (ms)'], 25, ec='black')
# plot acceptable ping
ax[0][0].axvline(acceptable_ping, color='yellow', linewidth=3)
ax[0][0].set_title('Ping (ms)', fontsize=16)

# plot upload speed
ax[0][1].hist(df_compact['Upload (Mbit/s)'], 25)
# plot acceptable upload
ax[0][1].axvline(acceptable_upload, color='red', linewidth=1)
ax[0][1].set_title('Upload (Mbit/s)', fontsize=16)
# plot download speed
ax[1][0].hist(df_compact['Download (Mbit/s)'], 25)

# plot acceptable download
ax[1][0].axvline(acceptable_download, color='red', linewidth=1)
ax[1][0].set_title('Download (Mbit/s)', fontsize=16)
#ax[1][1].set_visible(False)
ax[1][1].plot(t, df_compact['Ping (ms)'], label='Ping (ms)')
ax[1][1].axhline(acceptable_ping, color='red', linewidth=1)
ax[1][1].set_title('Ping (ms)', fontsize=16)
plt.show()


# compute the means and the standard deviations of the various measures
means = df_compact.mean()
stands = df_compact.std()

# this makes the results look better with labels
quote_ping = (means['Ping (ms)'], stands['Ping (ms)'])
quote_download = (means['Download (Mbit/s)'], stands['Download (Mbit/s)'])
quote_upload = (means['Upload (Mbit/s)'], stands['Upload (Mbit/s)'])

# print the results
print('Average ping time: {} ± {} ms'.format(*quote_ping))
# print Average download speed
print('Average ping time: {} ± {} ms'.format(*quote_ping))
print('Average download speed: {} ± {} Mbit/s'.format(*quote_download))
print('Average upload speed: {} ± {} Mbit/s'.format(*quote_upload))
# print Average upload speed
# ...
# blank line
print
print('Distance of acceptable Ping speed from average: {:.2f} standard deviations'.format((quote_ping[0]-acceptable_ping)/quote_ping[1]))
# print('Distance of acceptable Download speed from average
print('Distance of acceptable Download speed from average: {:.2f} standard deviations'.format((quote_download[0]-acceptable_download)/quote_download[1]))
# print('Distance of acceptable Upload speed from average
print('Distance of acceptable Upload speed from average: {:.2f} standard deviations'.format((quote_upload[0]-acceptable_upload)/quote_upload[1]))

print('{:.2f}% of measurements are lower than the acceptable download speed.'.format(np.sum(df_compact['Download (Mbit/s)'] < acceptable_download)/float(len(df_compact))*100))


print('{:.2f}% of measurements are higher than the acceptable ping speed.'.format(np.sum(df_compact['Ping (ms)']>acceptable_ping)/float(len(df_compact))*100))

print('{:.2f}% of measurements are lower than the acceptable upload speed.'.format(np.sum(df_compact['Upload (Mbit/s)']<acceptable_upload)/float(len(df_compact))*100))

all_three = np.sum((df_compact['Ping (ms)']>acceptable_ping) & (df_compact['Download (Mbit/s)']<acceptable_download) & (df_compact['Upload (Mbit/s)']<acceptable_upload))
print('{:.2f}% of measurements are not acceptable in three cases.'.format(all_three/float(len(df_compact))*100))

ping_upload = np.sum((df_compact['Ping (ms)']>acceptable_ping) & (df_compact['Upload (Mbit/s)']<acceptable_upload))
print('{:.2f}% of measurements are not acceptable for ping and upload.'.format(ping_upload/float(len(df_compact))*100))

#ping_download = ...
ping_download = np.sum((df_compact['Ping (ms)']>acceptable_ping) & (df_compact['Download (Mbit/s)']<acceptable_download))

#print(...)
print('{:.2f}% of measurements are not acceptable for ping and download.'.format(ping_download/float(len(df_compact))*100))

# upload_download = ...
upload_download = np.sum((df_compact['Download (Mbit/s)']<acceptable_download) & (df_compact['Upload (Mbit/s)']<acceptable_upload))

#print(...)
print('{:.2f}% of measurements are not acceptable for upload and download.'.format(upload_download/float(len(df_compact))*100))