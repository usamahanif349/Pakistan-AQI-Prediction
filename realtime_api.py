import requests
import streamlit as st

def get_api_token():
    """Reads the WAQI token from Streamlit secrets. Falls back to None if not configured."""
    try:
        return st.secrets["WAQI_TOKEN"]
    except Exception:
        return None

def pm25_to_us_aqi(pm25):
    if pm25 is None or pm25 == "N/A":
        return None
    try:
        c = float(pm25)
    except (ValueError, TypeError):
        return None

    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= c <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
            return round(aqi)

    if c > 500.4:
        return 500
    return 0


def get_live_city_aqi(city_name, city_lookup):
    api_token = get_api_token()
    if not api_token:
        return None

    meta = city_lookup.get(city_name, {})
    lat = meta.get("latitude")
    lon = meta.get("longitude")

    query_url = f"https://api.waqi.info/feed/{city_name.lower()}-pakistan/?token={api_token}"

    try:
        response = requests.get(query_url, timeout=5)
        data = response.json()

        if data.get("status") != "ok" and lat and lon:
            query_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={api_token}"
            response = requests.get(query_url, timeout=5)
            data = response.json()

        if data.get("status") == "ok":
            iaqi = data["data"].get("iaqi", {})
            raw_pm25 = iaqi.get("pm25", {}).get("v", None)
            api_aqi = data["data"].get("aqi", None)

            calculated_us_aqi = pm25_to_us_aqi(raw_pm25)
            final_aqi = calculated_us_aqi if calculated_us_aqi is not None else api_aqi

            time_str = data["data"].get("time", {}).get("s", "N/A")
            station_name = data["data"].get("city", {}).get("name", f"{city_name}, Pakistan")

            # Parse the 5-day forecast that WAQI already includes in this same response
            # (no extra API call needed) - purely additive, doesn't affect existing keys.
            forecast_list = []
            try:
                daily_pm25 = data["data"].get("forecast", {}).get("daily", {}).get("pm25", [])
                for day_entry in daily_pm25[:5]:
                    day_aqi = pm25_to_us_aqi(day_entry.get("avg"))
                    if day_aqi is not None:
                        forecast_list.append({"date": day_entry.get("day"), "aqi": day_aqi})
            except Exception:
                forecast_list = []

            return {
                "live_aqi": final_aqi,
                "pm25": raw_pm25 if raw_pm25 is not None else "N/A",
                "time": time_str,
                "station": station_name,
                "forecast": forecast_list
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None


def get_live_network_aqi(city_lookup):
    """Fetch live AQI for every city in city_lookup. Returns {city_name: live_data_or_None}.
    Purely additive helper - does not alter get_live_city_aqi's existing behavior or return shape."""
    results = {}
    for city_name in city_lookup.keys():
        results[city_name] = get_live_city_aqi(city_name, city_lookup)
    return results
def get_live_network_aqi(city_lookup):
    """Fetch live AQI for every city in city_lookup. Returns {city_name: live_data_or_None}.
    Purely additive helper - does not alter get_live_city_aqi's existing behavior or return shape."""
    results = {}
    for city_name in city_lookup.keys():
        results[city_name] = get_live_city_aqi(city_name, city_lookup)
    return results
