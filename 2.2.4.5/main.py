import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium

from pprint import pprint

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_path = './Data/Map-Crime_Incidents-Previous_Three_Months.csv'
    SF= pd.read_csv(dataset_path)
    pd.set_option('display.max_rows', 10)

    pprint(SF.columns)
    pprint(len(SF))



