from dataclasses import dataclass
from weatherkit import WeatherResponse 

#Task 3: Inside the Boundary

@dataclass
class HourlyReading:
    """One hour of weather. temperature_c in Celsius, precipitation_mm in millimeters."""
    timestamp: str
    temperature_c: float
    precipitation_mm: float
    
    
def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a validated columnar response into one reading per hour.

    Args:
        response: A validated WeatherResponse.

    Returns:
        One HourlyReading per hour, in the order returned by the API.
    """
    h = response.hourly
    return [
        HourlyReading(h.time[i], h.temperature_2m[i], h.precipitation[i])
        for i in range(len(h.time))
    ]
    
"""
Subjective comment: a good answer identifies where the boundary is. WeatherResponse describes outside data that 
arrives from the API and must be checked, so it is a Pydantic model. HourlyReading describes inside data that the 
student's own code produces from an already-validated response, so it is a dataclass. The data is validated once 
at the boundary, and validating it again inside would be wasted work.
"""