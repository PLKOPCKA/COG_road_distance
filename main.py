
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from joblib import Parallel, delayed
import time
import json

start = time.time()

solver_algo = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']
cogs = [1, 2, 3]
sensitivities = ['Base']
outputs = {}


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


def centroids(df_solve, num_centroids, algo):
    # define objective function
    def objective(c):
        sum_distance = 0
        for _, row in df_solve.iterrows():
            min_distance = float('inf')
            for i in range(num_centroids):
                distance = haversine_distance(row['lat'], row['lon'], c[i*2], c[i*2+1])*row['volume']
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

    # optimize centroids using method
    result = minimize(objective, c_0, method=algo)

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
            df_output = df_filter[['name', 'lat', 'lon', 'volume']]
            df_output['centroid'] = np.argmin([haversine_distance(df_output['lat'], df_output['lon'], c[0], c[1]) for c in centroid], axis = 0)
            df_output['distance'] = df_output.apply(lambda row: haversine_distance(row.lat, row.lon, centroid[int(row.centroid)][0], centroid[int(row.centroid)][1])*row.volume, axis=1)
            distance = df_output['distance'].sum()
            if min_distance > distance:
                min_distance = distance
                best_method = algo
                best_cogs = centroid

        outputs[scenario][i][best_method] = best_cogs



# Initialize empty lists to store the data
base_list = []
id_list = []
cg1_list = []
cg2_list = []

# Loop through the dictionary and extract the data
for base, values in outputs["Base"].items():
    for id, cg in values.items():
        base_list.extend([base] * len(cg))
        id_list.extend([id] * len(cg))
        cg1_list.extend([x[0] for x in cg])
        cg2_list.extend([x[1] for x in cg])

# Create a dataframe from the lists
df = pd.DataFrame({"Base": base_list, "ID": id_list, "CG1": cg1_list, "CG2": cg2_list})


# Save DataFrame to a CSV file
df.to_csv('COG_Output.csv', index=False)

# with open("outputs.txt", "w") as outfile:
#     # Write the dictionary to the file
#     json.dump(outputs, outfile)

end = time.time()
print(end - start)