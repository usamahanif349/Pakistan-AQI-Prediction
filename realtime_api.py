import os
import requests

WAQI_API_BASE_URL = "https://api.waqi.info/feed"

def get_live_city_aqi(city_name: str, api_token: str = "demo"):
    """
    Fetches real-time AQI and pollutant data for a given city.
    Returns None if the network request fails.
    """
    try:
        url = f"{WAQI_API_BASE_URL}/{city_name}/?token={api_token}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                payload = data.get("data", {})
                iaqi = payload.get("iaqi", {})
                
                return {
                    "city": city_name,
                    "live_aqi": payload.get("aqi"),
                    "dominant_pollutant": payload.get("dominentpol"),
                    "pm25": iaqi.get("pm25", {}).get("v"),
                    "pm10": iaqi.get("pm10", {}).get("v"),
                    "temperature": iaqi.get("t", {}).get("v"),
                    "humidity": iaqi.get("h", {}).get("v"),
                    "time": payload.get("time", {}).get("s")
                }
    except Exception as e:
        print(f"[API Warning] Failed to fetch live data for {city_name}: {e}")
        return None

    return None
