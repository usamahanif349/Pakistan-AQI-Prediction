import requests

API_TOKEN = "956d68b497aaec4796bab052547399e7e014bff1"

def pm25_to_us_aqi(pm25):
    """
    Converts PM2.5 concentration (µg/m³) into standard US EPA AQI using official breakpoints.
    """
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
    """
    Fetches real-time PM2.5 data from WAQI API and computes standard US AQI.
    """
    meta = city_lookup.get(city_name, {})
    lat = meta.get("latitude")
    lon = meta.get("longitude")

    if lat and lon:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={API_TOKEN}"
    else:
        url = f"https://api.waqi.info/feed/{city_name}/?token={API_TOKEN}"

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get("status") == "ok":
            iaqi = data["data"].get("iaqi", {})
            raw_pm25 = iaqi.get("pm25", {}).get("v", None)

            # Get raw station AQI provided by WAQI feed
            api_aqi = data["data"].get("aqi", None)

            # Convert PM2.5 concentration to EPA US AQI
            calculated_us_aqi = pm25_to_us_aqi(raw_pm25)

            # Fallback to API AQI if PM2.5 calculation is unavailable
            final_aqi = calculated_us_aqi if calculated_us_aqi is not None else api_aqi

            time_str = data["data"].get("time", {}).get("s", "N/A")
            station_name = data["data"].get("city", {}).get("name", city_name)

            return {
                "live_aqi": final_aqi,
                "pm25": raw_pm25 if raw_pm25 is not None else "N/A",
                "time": time_str,
                "station": station_name
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None
