import geopy
import pandas as pd

df = pd.read_csv("data/AGENCY_A/base/para_transit_trips_2021.csv")


def get_zipcode(_df, _geo_locator, lat_field, lon_field):
    location = _geo_locator.reverse((_df[lat_field], _df[lon_field]))
    try:
        post_code = location.raw['address']['postcode']
    except KeyError:
        post_code = -1
    return post_code


geolocator = geopy.Nominatim(user_agent='test')

df["Pickup ZIP"] = df.apply(
    get_zipcode, axis=1, _geo_locator=geolocator, lat_field='Pickup LAT', lon_field='Pickup LON'
)
df["Dropoff ZIP"] = df.apply(
    get_zipcode, axis=1, _geo_locator=geolocator, lat_field='Dropoff LAT', lon_field='Dropoff LON'
)

df = df[[
    "bookingid", "Sch Time in HH:MM:SS", "ldate",
    "Pickup LAT", "Pickup LON", "Pickup ZIP", "Dropoff LAT", "Dropoff LON", "Dropoff ZIP",
    "Passenger Types", "AM/WC",
]]
df.to_csv("data/AGENCY_A/base/para_transit_trips_2021.csv", index=False)
