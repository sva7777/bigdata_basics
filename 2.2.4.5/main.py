import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import folium

from pprint import pprint

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_path = './Data/Map-Crime_Incidents-Previous_Three_Months.csv'
    SF= pd.read_csv(dataset_path)
    pd.set_option('display.max_rows', 10)

    pprint(SF.columns)
    pprint(len(SF))

    SF['Month'] = SF['Date'].apply(lambda row: int(row[0:2]))
    SF['Day'] = SF['Date'].apply(lambda row: int(row[3:5]))

    print(SF['Month'][0:2])
    print(SF['Day'][0:2])
    print(type(SF['Month'][0]))

    del SF['IncidntNum']
    SF.drop('Location', axis=1, inplace=True)

    print(SF.columns)

    CountCategory = SF['Category'].value_counts(ascending=True)
    print(CountCategory)

    print(SF['PdDistrict'].value_counts(ascending=True))

    AugustCrimes = SF[SF['Month'] == 8]
    print(AugustCrimes)

    AugustCrimes = SF[SF['Month'] == 8]
    AugustCrimesB = SF[SF['Category'] == 'BURGLARY']
    print(len(AugustCrimesB))

    Crime0704 = SF.query('Month == 7 and Day == 4')
    print(Crime0704)

    plt.plot(SF['X'], SF['Y'], 'ro')
    plt.show()

    pd_districts = np.unique(SF['PdDistrict'])
    pd_districts_levels = dict(zip(pd_districts, range(len(pd_districts))))
    print(pd_districts_levels)

    SF['PdDistrictCode'] = SF['PdDistrict'].apply(lambda row: pd_districts_levels[row])

    plt.scatter(SF['X'], SF['Y'], c=SF['PdDistrictCode'])
    plt.show()

    districts = np.unique(SF['PdDistrict'])
    print(list(colors.cnames.values())[0:len(districts)])

    color_dict = dict(zip(districts, list(colors.cnames.values())[0:-1:len(districts)]))
    print(color_dict)

    map_osm = folium.Map(location=[SF['Y'].mean(), SF['X'].mean()], zoom_start=12)
    plotEvery = 50
    obs = list(zip(SF['Y'], SF['X'], SF['PdDistrict']))

    for el in obs[0:-1:plotEvery]:
        folium.CircleMarker(el[0:2], color=color_dict[el[2]], fill_color=el[2], radius=10).add_to(map_osm)

    print(map_osm)
