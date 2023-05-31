
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from joblib import Parallel, delayed
import time
import json
import requests
import time

start = time.time()

solver_algo = ['Nelder-Mead']
cogs = [1]
sensitivities = ['Base']
outputs = {}

calls = {}
df = pd.read_csv("C:/Users/PLKOPCKA/Documents/COG.csv")

X_u = df['lat'].max()
X_l = df['lat'].min()
Y_u = df['lon'].max()
Y_l = df['lon'].min()


def initial_points(num_centroids, df_arg):
    kmeans = KMeans(num_centroids, random_state=0).fit(df_arg.loc[df_arg['volume'] > 0, ['lat', 'lon']],
                  sample_weight=df_arg.loc[df_arg['volume'] > 0, 'volume'])

    # Get centers of gravity from K-means
    cogs = kmeans.cluster_centers_
    return cogs


def haversine_distance(lat1, lon1, lat2, lon2, circuity=1.17, earth_radius=6371):
    # Convert latitude and longitude to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Calculate the difference between the two points
    lat_diff = lat2 - lat1
    lon_diff = lon2 - lon1

    # Use the haversine formula to calculate the distance
    a = np.sin(lat_diff / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lon_diff / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Return the distance in kilometers
    return c * earth_radius * circuity


def road_distance(lat1, lon1, lat2, lon2):
    api_key = 'Ajkfo_OAlIzlY_N-tv0jGylQ69AUpvO9Y7YopcohNMLfvnFAa1di5EaB8zXslrs0'

    origins = str(lat1) + ', ' + str(lon1)
    destinations = str(lat2) + ', ' + str(lon2)
    dest = origins+destinations[:9]

    if dest in calls:
        distance = calls[dest]
    else:
        url = f'https://dev.virtualearth.net/REST/v1/Routes/DistanceMatrix?origins={origins}&destinations={destinations}&travelMode=driving&key={api_key}'
        response = requests.get(url)
        data = response.json()
        distance = data['resourceSets'][0]['resources'][0]['results'][0]['travelDistance']*1.60934
        calls[dest] = distance

    if distance < 0 or not distance:
        distance = 1000
    return distance


def road_time_truck(lat1, lon1, lat2, lon2):
    api_key = 'Ajkfo_OAlIzlY_N-tv0jGylQ69AUpvO9Y7YopcohNMLfvnFAa1di5EaB8zXslrs0'

    origins = str(lat1) + ', ' + str(lon1)
    destinations = str(lat2) + ', ' + str(lon2)
    dest = origins + destinations[:9]

    if dest in calls:
        travel_duration = calls[dest]
    else:
        url = f'https://dev.virtualearth.net/REST/v1/Routes?wp.0={origins}&wp.1={destinations}&travelMode=truck&key={api_key}&routePath=True'
        response = requests.get(url)
        data = response.json()
        travel_duration = data['resourceSets'][0]['resources'][0]['travelDuration'] / 3600
        calls[dest] = travel_duration
    return travel_duration


def centroids(df_solve, num_centroids, algo):
    # define objective function
    def objective(c):
        sum_distance = 0
        for _, row in df_solve.iterrows():
            min_distance = float('inf')
            for i in range(num_centroids):
                distance = road_distance(row['lat'], row['lon'], c[i*2], c[i*2+1])*row['volume']
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
        return sum_distance

    # initial guess for centroids
    points = []
    k_output = initial_points(num_centroids, df_solve).tolist()
    for el in k_output:
      for el2 in el:
        points.append(el2)
    c_0 = points

    bounds = [(X_l, X_u), (Y_l, Y_u)]
    # optimize centroids using method
    result = minimize(objective, c_0, method=algo, bounds=bounds)

    # return optimal centroids
    return result.x.reshape(-1, 2)


for scenario in sensitivities:
    df_filter = df[df['sensitivity'] == scenario]
    outputs[scenario] = {}
    for i in cogs:
        outputs[scenario][i] = {}
        min_distance = float('inf')
        for algo in solver_algo:
            centroid = centroids(df_filter, i, algo).tolist()
            best_method = algo
            best_cogs = centroid

        outputs[scenario][i][best_method] = best_cogs


with open("outputs.txt", "w") as outfile:
    # Write the dictionary to the file
    json.dump(outputs, outfile)


end = time.time()
print(end - start)