from IPython.display import clear_output
import numpy as np


def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))
    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


def geo_location_index(lat, lon, grid_size=0.3):
    lat = np.array(lat)
    lon = np.array(lon)

    cir_equator = 40070.  # Equatorial circumference in km
    cir_pole = 39931.  # Polar circumference in km
    km_per_deg_equator = cir_equator / 360.0  # km distance per degree
    km_per_deg_pole = cir_pole / 360.0  # km distance per degree

    lat_index = np.round(lat * km_per_deg_pole / float(grid_size))
    lon_index = (np.cos(np.deg2rad(lat)) * km_per_deg_equator) * lon
    lon_index = np.round(lon_index / float(grid_size))

    lon_max = km_per_deg_equator * 180.0 / float(grid_size)
    lon_min = km_per_deg_equator * -180.0 / float(grid_size)
    lat_min = km_per_deg_pole * -90.0 / float(grid_size)

    lat_index = lat_index - lat_min
    lon_index = lon_index - lon_min
    lon_max = lon_max - lon_min

    location_index = np.round(lat_index * (lon_max + 1) + lon_index)
    return location_index.astype('int64')
