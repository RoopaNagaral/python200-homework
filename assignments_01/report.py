import json
from pathlib import Path

from weatherkit.schemas import WeatherResponse
from weatherkit.summarize import DailyAggregator
from weatherkit.records import HourlyReading, to_readings


def main() -> None:
    json_path = Path(__file__).parent / "weather_raw.json"
    with open(json_path, "r") as f:
        raw = json.load(f)

    data = WeatherResponse.model_validate(raw)

    # Task 2
    print("Latitude:", data.latitude)
    print("Timezone:", data.timezone)
    print("Hourly observations:", len(data.hourly.time))

    #Task 4
    #Build HourlyReading objects from the columnar hourly block
    readings = to_readings(data)

    #Aggregate to daily summaries
    aggregator = DailyAggregator()  # min_hours=24 by default
    summaries = aggregator.summarize(readings)

    print("\nNumber of summaries:", len(summaries))
        
    #Task 6 -
    # 1. Validate API response
        #Validating API response is done in task 2
        
    # 2. Convert columnar hourly block → HourlyReading objects
        #Converting columnar hourly block → HourlyReading objects is done task 4
        
    # 3. Aggregate into daily summaries
    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    # 4. Print readable table
    print("\nDaily Weather Summary\n")
    print(f"{'Date':<12} {'High':>6} {'Low':>6} {'Precip':>10} {'Range':>8}")
    print("-" * 46)

    for s in summaries:
        print(
            f"{s.date:<12} "
            f"{s.temp_max:>6.1f} "
            f"{s.temp_min:>6.1f} "
            f"{s.precipitation_sum:>10.2f} "
            f"{s.temp_range():>8.1f}"
        )

    # 5. Print warning about incomplete days
    if incomplete:
        print("\nWARNING: Incomplete days dropped:", ", ".join(incomplete))
        
if __name__ == "__main__":
    main()
    

"""
Subjective comment: If this guard were omitted, then importing report.py from another module
(for example, to reuse main() or helper functions) would immediately run
the script — loading files, validating JSON, printing output — even though
the caller only wanted to import a function. The guard prevents accidental
execution on import.
"""

"""
Task 7 : Reflection
____________________________

1. Rejecting the whole file on a single null temperature.A good answer names one situation where strict rejection 
is right (for example, a training dataset where a silent NaN would corrupt a model, so failing loudly at the 
boundary is safer) and one where tolerance is better (for example, a monitoring dashboard that should 
still show 167 good hours rather than nothing). The schema change to tolerate the gap is widening the 
type to list[float | None], which records the decision to tolerate missing values in the schema itself.

2. min_hours=24 when a pipeline runs at noon. A good answer explains that a day only half over has roughly 12 hours 
of data, so its "daily maximum" is really a half-day maximum and cannot be trusted. incomplete_days() helps 
because the partial day is dropped from the summaries but its date is still reported, so the gap is visible 
rather than silent.

3. Package rather than one file, and Week 10. A good answer names one concrete benefit, such as a pipeline being 
able to import weatherkit and call its classes directly, testing the package in isolation, or reorganizing 
internal files without breaking callers because the public interface is defined in __init__.py.

"""
