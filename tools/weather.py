# tools/weather.py
import httpx
import json
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class WeatherInput(BaseModel):
    city: str = Field(description="Название города на английском, например: Moscow, London")
    forecast_days: int = Field(
        default=1,
        ge=1,
        le=7,
        description="Количество дней прогноза (1-7). 1 = только текущая погода"
    )


@tool
def get_weather_tool(city: str, forecast_days: int = 1) -> str:
    """
    Получает погоду и прогноз для города через Open-Meteo API.
    
    Args:
        city: Название города на английском
        forecast_days: Дней прогноза (1-7), по умолчанию 1
    
    Returns:
        JSON-строка с текущей погодой и/или прогнозом
    """
    try:
        with httpx.Client(timeout=10.0) as client:
            # 🔹 Шаг 1: Геокодинг
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_response = client.get(
                geo_url,
                params={"name": city, "count": 1, "language": "en", "format": "json"}
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            
            if not geo_data.get("results"):
                return json.dumps({
                    "error": f"Город '{city}' не найден. Укажите название на английском."
                }, ensure_ascii=False)
            
            location = geo_data["results"][0]
            lat, lon = location["latitude"], location["longitude"]
            city_name = location.get("name", city)
            country = location.get("country", "")
            
            # 🔹 Шаг 2: Формируем запрос к weather API
            weather_url = "https://api.open-meteo.com/v1/forecast"
            
            # Базовые параметры
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
                "timezone": "auto",
            }
            
            # Добавляем прогноз, если нужно больше 1 дня
            if forecast_days > 1:
                params["daily"] = "weather_code,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,precipitation_probability_max"
                params["forecast_days"] = forecast_days
            
            weather_response = client.get(weather_url, params=params)
            weather_response.raise_for_status()
            weather_data = weather_response.json()
            
            # 🔹 Шаг 3: Парсим ответ
            current = weather_data.get("current", {})
            if not current:
                return json.dumps({"error": "Нет данных о текущей погоде"}, ensure_ascii=False)
            
            # Декодер WMO-кодов
            weather_codes = {
                0: "ясно", 1: "преимущественно ясно", 2: "переменная облачность",
                3: "пасмурно", 45: "туман", 48: "иней",
                51: "лёгкая морось", 53: "морось", 55: "сильная морось",
                61: "слабый дождь", 63: "дождь", 65: "сильный дождь",
                71: "слабый снег", 73: "снег", 75: "сильный снег",
                77: "снежные зёрна", 80: "лёгкий ливень", 81: "ливень", 82: "сильный ливень",
                95: "гроза", 96: "гроза с градом", 99: "сильная гроза с градом"
            }
            
            # Текущая погода
            result = {
                "location": f"{city_name}{', ' + country if country else ''}",
                "current": {
                    "time": current.get("time", ""),
                    "temperature": f"{current.get('temperature_2m', 'N/A')}°C",
                    "condition": weather_codes.get(current.get('weather_code'), "неизвестно"),
                    "wind": f"{current.get('wind_speed_10m', 'N/A')} м/с",
                    "humidity": f"{current.get('relative_humidity_2m', 'N/A')}%"
                }
            }
            
            # 🔹 Прогноз по дням (если запрошено)
            if forecast_days > 1 and "daily" in weather_data:
                daily = weather_data["daily"]
                forecasts = []
                
                for i in range(min(forecast_days, len(daily.get("time", [])))):
                    day_forecast = {
                        "date": daily["time"][i],
                        "condition": weather_codes.get(daily["weather_code"][i], "неизвестно"),
                        "temp_max": f"{daily['temperature_2m_max'][i]}°C",
                        "temp_min": f"{daily['temperature_2m_min'][i]}°C",
                        "wind_max": f"{daily['wind_speed_10m_max'][i]} м/с",
                        "precip_prob": f"{daily['precipitation_probability_max'][i]}%"
                    }
                    forecasts.append(day_forecast)
                
                result["forecast"] = forecasts
            
            return json.dumps(result, ensure_ascii=False)
            
    except httpx.TimeoutException:
        return json.dumps({"error": "Таймаут при запросе к погодному сервису"}, ensure_ascii=False)
    except httpx.RequestError as e:
        return json.dumps({"error": f"Ошибка сети: {str(e)}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Ошибка: {str(e)}"}, ensure_ascii=False)