from smolagents import Tool
import requests
import os
from typing import Dict, Any

class WeatherTool(Tool):

    name ="get_weather"
    description ="Fetch live weather for a given city using openweather api."
    inputs ={
        "city":{
           "type":"string",
           "description":"the city name(e.g. london,madurai,etc..,).",
        },
        "units":{
           "type": "string",
           "description": "Units for temperature: 'metric'=Celsius, 'imperial'=Fahrenheit, 'standard'=Kelvin",
           "nullable": True
        }
    }
    output_type ="object" 

    def forward(self,city:str,units:str ="metric"):

        api_key = os.getenv["WEATHER_API_KEY"]

        if not api_key:

            return{"error":"api is not found "}
        
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units={units}"

        try:

            res = requests.get(url,timeout=10).json()

        except Exception as e:

            return {"error": f"Request failed: {str(e)}"}
        
        if res.get("cod") != 200:

            return{"error": res.get("message", "Unknown error")}
        
        return {
            "city": res["name"],
            "temperature": res["main"]["temp"],
            "description": res["weather"][0]["description"],
            "humidity": res["main"]["humidity"],
            "wind_speed": res["wind"]["speed"],
        }
def weather_tool(city: str, units: str = "metric") -> Dict:
    """
    Fetch live weather for a given city using OpenWeather API.

    Args:
        city (str): The city name (e.g. London, Madurai, etc.).
        units (str, optional): Units for temperature.
            - 'metric' = Celsius
            - 'imperial' = Fahrenheit
            - 'standard' = Kelvin

    Returns:
        dict: Weather information including city, temperature, description, humidity, wind speed.
    """
    tool = WeatherTool()
    return tool.forward(city, units)


# if __name__ == "__main__":
#     print(weather_tool("Madurai"))

     
