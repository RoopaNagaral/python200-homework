import json
from pathlib import Path
from dataclasses import dataclass
import pytest

# Q1
class ModelInfo:
    def __init__(self,path):
        with open(path) as f:
            self.data = json.load(f)
            
    def feature_count(self) -> int:
        return len(self.data["features"])
    
# Create a small JSON file
sample_data = {
    "model_name": "weather_classifier",
    "features": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
}

model_folder = Path(__file__).parent / "models"
model_path = model_folder / "model_info.json"

with open(model_path, "w") as file:
    json.dump(sample_data, file)

# Build a ModelInfo object
model = ModelInfo(model_path)

# Print the feature count
print("Feature count:", model.feature_count())

# Loading the file in __init__ is better because the file is read only once,
# avoiding repeated disk access and improving performance when feature_count()
# is called many times.

# Q2 
@dataclass
class DayForecast:
     date : str
     high_c : float
     will_rain : bool
     
     def summary(self) -> str:
        rain = "rain expected" if self.will_rain else "no rain"
        return f"{self.date}: high {self.high_c}C, {rain}"
    
dayforcast1 = DayForecast("2023-06-15",24.1, "rain expected")
dayforcast2 = DayForecast("2023-06-16",26.1, "")

print("\nDay One forecast:", dayforcast1.summary())
print("Day Two forecast:", dayforcast2.summary())

# Q3
def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"temperature must be a number, got {text!r}")


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")
        
# Using pytest.raises(ValueError) without match would only check that a
# ValueError occurred. It would not catch an incorrect or unhelpful error
# message, such as "bad input" or "conversion failed", that does not
# mention "temperature".
