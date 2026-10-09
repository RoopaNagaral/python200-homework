from weatherkit.schemas import HourlyBlock, WeatherResponse
from weatherkit.records import HourlyReading
import weatherkit.records as wr
import pytest

def test_to_readings_returns_one_reading_per_hour_in_order():
    
    hourly_block = WeatherResponse.model_validate(
        {
                "latitude": 60.0,
                "longitude": 0.0,
                "timezone": "UTC",
                "elevation": 0.0,
                "hourly": {
                    "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
                    "temperature_2m": [10.0, 11.0],
                    "precipitation": [0.0, 0.1],
                }
        }
    )
    readings = wr.to_readings(hourly_block)
    assert len(readings) == 2
    assert readings[0].timestamp == "2026-01-01T00:00"
    assert readings[-1].timestamp == "2026-01-01T01:00"
    
def test_reading_values_match_input_indices():
    hourly_block = WeatherResponse.model_validate(
        {
            "latitude": 60.0,
            "longitude": 0.0,
            "timezone": "UTC",
            "elevation": 0.0,
            "hourly": {
                "time": ["2026-01-01T00:00", "2026-01-01T01:00", "2026-01-01T02:00"],
                "temperature_2m": [10.0, 11.0, 12.0],
                "precipitation": [0.0, 0.1, 0.2],
            }
        }
    )
    readings = wr.to_readings(hourly_block)
    
    for i, r in enumerate(readings):
        assert r.timestamp == hourly_block.hourly.time[i]
        assert r.temperature_c == hourly_block.hourly.temperature_2m[i]
        assert r.precipitation_mm == hourly_block.hourly.precipitation[i]
        
def test_hourly_readings_with_identical_fields_compare_equal():
    r1 = HourlyReading(
        timestamp="2026-01-01T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )
    r2 = HourlyReading(
        timestamp="2026-01-01T00:00",
        temperature_c=10.0,
        precipitation_mm=0.0,
    )

    assert r1 == r2