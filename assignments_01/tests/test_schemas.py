from weatherkit.schemas import HourlyBlock, WeatherResponse
import json
from pathlib import Path
import pytest
from pydantic import ValidationError

json_path = Path(__file__).parent.parent / "weather_raw.json"
with open(json_path, "r") as f:
    raw = json.load(f)

data = WeatherResponse.model_validate(raw)

def test_validate_respose_has_168_hours():
    assert len(data.hourly.time) == 168
    
def test_latitude_out_of_range_raises_validation_error():
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(
            {
                "latitude": 200.0,
                "longitude": 0.0,
                "timezone": "UTC",
                "elevation": 0.0,
                "hourly": {
                    "time": ["2026-01-01T00:00"],
                    "temperature_2m": [10.0],
                    "precipitation": [0.0],
                },
            }
        )


def test_mismatched_list_lengths_raise_validation_error():
    with pytest.raises(ValidationError):
        HourlyBlock.model_validate(
            {
                "time": ["2026-01-01T00:00", "2026-01-01T01:00", "2026-01-01T02:00"],
                "temperature_2m": [10.0, 11.0],  # one element short
                "precipitation": [0.0, 0.1, 0.0],
            }
        )


def test_null_in_temperature_raises_validation_error():
    with pytest.raises(ValidationError):
        HourlyBlock.model_validate(
            {
                "time": ["2026-01-01T00:00"],
                "temperature_2m": [None],
                "precipitation": [0.0],
            }
        )