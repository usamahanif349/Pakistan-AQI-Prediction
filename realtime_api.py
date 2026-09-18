import requests

API_TOKEN = "956d68b497aaec4796bab052547399e7e014bff1"

def get_live_city_aqi(city_name, city_lookup):
    """
    Fetches real-time AQI and PM2.5 data using city coordinates
    to ensure live readings from active 2026 sensors.
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
            aqi = data["data"]["aqi"]
            iaqi = data["data"].get("iaqi", {})
            pm25 = iaqi.get("pm25", {}).get("v", "N/A")
            time_str = data["data"].get("time", {}).get("s", "N/A")
            
            return {
                "live_aqi": aqi,
                "pm25": pm25,
                "time": time_str
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching live data: {e}")
        return None
