"""
Weather Tools - Fetch weather information using OpenWeatherMap API.
"""
import os
import requests
from core.tool_system import registry


@registry.register(
    name="get_weather",
    description="Get current weather information for a specified location",
    parameters={
        "type": "object",
        "required": ["location"],
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, or city name with country code (e.g., 'London', 'London,UK', 'New York,US')"
            },
            "units": {
                "type": "string",
                "description": "Units of measurement: 'metric' (Celsius), 'imperial' (Fahrenheit), or 'standard' (Kelvin)",
                "enum": ["metric", "imperial", "standard"],
                "default": "metric"
            }
        }
    }
)
def get_weather(location: str, units: str = "metric") -> dict:
    """
    Fetch current weather information for a location using OpenWeatherMap API.
    
    Args:
        location: City name or city name with country code
        units: Units of measurement (metric, imperial, or standard)
        
    Returns:
        A dictionary containing weather information or error details
    """
    try:
        # Get API key from environment
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            return {
                "error": "OPENWEATHERMAP_API_KEY environment variable not set"
            }
        
        # Build API request
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": api_key,
            "units": units
        }
        
        # Make API request
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract and format weather data
        unit_symbol = "°C" if units == "metric" else ("°F" if units == "imperial" else "K")
        
        weather_info = {
            "location": data["name"],
            "country": data["sys"]["country"],
            "temperature": f"{data['main']['temp']}{unit_symbol}",
            "feels_like": f"{data['main']['feels_like']}{unit_symbol}",
            "temp_min": f"{data['main']['temp_min']}{unit_symbol}",
            "temp_max": f"{data['main']['temp_max']}{unit_symbol}",
            "humidity": f"{data['main']['humidity']}%",
            "pressure": f"{data['main']['pressure']} hPa",
            "description": data["weather"][0]["description"],
            "weather_main": data["weather"][0]["main"],
            "wind_speed": data["wind"]["speed"],
            "clouds": f"{data['clouds']['all']}%",
            "coordinates": {
                "lat": data["coord"]["lat"],
                "lon": data["coord"]["lon"]
            }
        }
        
        # Add optional fields if available
        if "visibility" in data:
            weather_info["visibility"] = f"{data['visibility']} meters"
        
        if "wind" in data and "deg" in data["wind"]:
            weather_info["wind_direction"] = f"{data['wind']['deg']}°"
        
        return weather_info
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {
                "error": f"Location '{location}' not found"
            }
        return {
            "error": f"HTTP error: {str(e)}"
        }
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Request failed: {str(e)}"
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}"
        }