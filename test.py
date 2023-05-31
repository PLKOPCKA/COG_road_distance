import requests


def road_time_truck(lat1, lon1, lat2, lon2):
    api_key = 'Ajkfo_OAlIzlY_N-tv0jGylQ69AUpvO9Y7YopcohNMLfvnFAa1di5EaB8zXslrs0'

    origins = str(lat1) + ', ' + str(lon1)
    destinations = str(lat2) + ', ' + str(lon2)

    url = f'https://dev.virtualearth.net/REST/v1/Routes?wp.0={origins}&wp.1={destinations}&travelMode=truck&key={api_key}&routePath=True'
    response = requests.get(url)
    data = response.json()
    travel_duration = data['resourceSets'][0]['resources'][0]['travelDuration'] / 3600
    return travel_duration


def road_time_car(lat1, lon1, lat2, lon2):
    api_key = 'Ajkfo_OAlIzlY_N-tv0jGylQ69AUpvO9Y7YopcohNMLfvnFAa1di5EaB8zXslrs0'

    origins = str(lat1) + ', ' + str(lon1)
    destinations = str(lat2) + ', ' + str(lon2)

    url = f'https://dev.virtualearth.net/REST/v1/Routes/DistanceMatrix?origins={origins}&destinations={destinations}&travelMode=driving&key={api_key}'
    response = requests.get(url)
    data = response.json()
    travel_duration = data['resourceSets'][0]['resources'][0]['results'][0]['travelDuration'] / 60
    return travel_duration


print(road_time_truck('19.96566608','102.2435261','20.4146522','104.0485836'))
print(road_time_car('19.96566608','102.2435261','20.4146522','104.0485836'))

