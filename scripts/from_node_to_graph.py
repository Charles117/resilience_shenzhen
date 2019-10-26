import math
import numpy as np
import pandas as pd

def calculate_dist(lat_1, long_1, lat_2, long_2):
    tmp = 2 * math.pi * 6378.137 / 360
    north_south = (lat_1 - lat_2) * tmp
    lat = (lat_1 + lat_2) / 2
    east_west = (long_1 - long_2) * tmp * math.cos(lat / 180 * math.pi)
    return np.sqrt(east_west ** 2 + north_south ** 2)

filename = '../data/sensor_graph/graph_sensor_locations.csv' # replace input filename here
location_df = pd.read_csv(filename, dtype={'sensor_id': 'str'})

output_filename = '../data/sensor_graph/custom_distance.csv' # replace output filename here
f = open(output_filename, 'w')
f.write('from,to,cost\n')
f.close()

f = open(output_filename, 'a')
for i in range(len(location_df)):
    for j in range(i, len(location_df)):
        u = location_df.iloc[i]
        v = location_df.iloc[j]
        dist = calculate_dist(u['latitude'], u['longitude'], v['latitude'], v['longitude'])
        u = u['sensor_id']
        v = v['sensor_id']
        print(u, v, dist)
        f.write(f"{u},{v},{dist}\n")
f.close()


