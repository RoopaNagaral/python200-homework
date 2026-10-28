from weather_model import WeatherClassifier
from pathlib import Path

# Task 5: The Script
def main() -> None:
    """Load the saved model and make predictions for sample days."""
    model_folder = Path(__file__).parent / "models"
    model_path = model_folder / "weather_classifier.pkl"
    classifier = WeatherClassifier(model_path)

    days = [
        # Clearly good day
        {
            "temperature_2m_max": 22,
            "temperature_2m_min": 10,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 10,
        },
        # Clearly bad day
        {
            "temperature_2m_max": -3,
            "temperature_2m_min": -10,
            "precipitation_sum": 12.0,
            "wind_speed_10m_max": 40,
        },
        # Borderline day: a little cool, light rain
        {
            "temperature_2m_max": 8,
            "temperature_2m_min": 1,
            "precipitation_sum": 2.5,
            "wind_speed_10m_max": 18,
        },
        # Borderline day: warm but windy
        {
            "temperature_2m_max": 25,
            "temperature_2m_min": 15,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 28,
        },
    ]

    predictions = classifier.predict(days)

    for i, (day, prediction) in enumerate(zip(days, predictions), start=1):
        print(
            f"Day {i}: "
            f"max={day['temperature_2m_max']}°C, "
            f"min={day['temperature_2m_min']}°C, "
            f"precip={day['precipitation_sum']} mm, "
            f"wind={day['wind_speed_10m_max']} km/h "
            f"-> label={prediction.label}, "
            f"probability={prediction.probability:.2f}"
        )


# Without this guard, the code in main() would run immediately whenever
# another script imported predict_weather.py. That would cause unwanted
# predictions and output even if the importing script only wanted to
# reuse a helper function or class from this file.
if __name__ == "__main__":
    main()
    
# Task 6: Reflection
# Reflection
#
# 1. Why select features in a fixed order?
#    Selecting features in a fixed order prevents data from being matched to
#    the wrong columns. For example, if temperature_2m_max and
#    wind_speed_10m_max were accidentally swapped, the model might interpret
#    a temperature of 25°C as a wind speed of 25 km/h and a wind speed of
#    10 km/h as a temperature of 10°C. This could produce incorrect
#    predictions even though all required values were provided.
#
# 2. Why do the tests build their own model instead of loading the real file?
#    Tests should be independent of external files. If the real
#    models/weather_classifier.pkl file were missing, renamed, corrupted, or
#    replaced with a different model, the tests could fail for reasons
#    unrelated to the code being tested. Building a tiny model inside a
#    fixture makes the tests reproducible, self-contained, and reliable on
#    any machine.
#
# 3. Readiness for a Week 10 pipeline:
#    One thing that makes this component ready is that predict() accepts a
#    batch of weather records and returns predictions in the same order, which
#    works well for database-driven workflows.
#
#    One thing I would add before production is logging and additional input
#    validation. For example, I would log prediction requests and results and
#    check for invalid values such as negative precipitation or missing data
#    before making predictions.