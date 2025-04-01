import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # radius of the Earth
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def find_closest_bus_station(df, current_lat, current_lon):
    df["distance"] = df.apply(
        lambda row: haversine(
            current_lat, current_lon, row["Latitude"], row["Longitude"]
        ),
        axis=1,
    )
    closest_point = df.loc[df["distance"].idxmin()]

    return closest_point

def hex_to_rgb(hex_value):
    hex_value = hex_value.lstrip('#')
    return list(int(hex_value[i:i+2], 16) for i in (0, 2, 4))

if __name__ == '__main__':
    print(hex_to_rgb("#ff0000"))