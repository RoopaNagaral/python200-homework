from dataclasses import dataclass, Field
import joblib
from pathlib import Path
import os
from typing import Dict, List, Any
import pandas as pd


# Task 1: The Prediction Dataclass

@dataclass
class Prediction:
    """
        Represents a model prediction for whether a day is good for running.

        Attributes:
            label (str): Either "good" or "skip".
            probability (float): Probability from 0.0 to 1.0 that the day is
                good for running.
    """
    label : str 
    probability : float
    

# Task 2: The WeatherClassifier Class

class WeatherClassifier:
    """
    Loads a trained weather-classification model and uses it to make
    predictions for future days.
    """

    REQUIRED_FEATURES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    def __init__(self, model_path: str) -> None:
        """
        Initialize the classifier from a saved model file.

        Args:
            model_path (str): Path to a saved .pkl model file.

        Raises:
            FileNotFoundError: If the specified model file does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        self.model = joblib.load(model_path)

    def predict(self, days: List[Dict[str, Any]]) -> List[Prediction]:
        """
        Predict whether each day is good for running.

        Args:
            days (List[Dict[str, Any]]): A list of dictionaries containing
                weather data. Each dictionary must include:
                temperature_2m_max,
                temperature_2m_min,
                precipitation_sum,
                wind_speed_10m_max

        Returns:
            List[Prediction]: Predictions in the same order as the input.

        Raises:
            ValueError: If any required feature is missing from an input day.
        """
        if not days:
            return []

        for i, day in enumerate(days):
            missing = [
                feature
                for feature in self.REQUIRED_FEATURES
                if feature not in day
            ]

            if missing:
                raise ValueError(
                    f"Day {i} is missing required features: "
                    f"{', '.join(missing)}"
                )

        # Build feature matrix in a fixed order expected by the model.
        X = pd.DataFrame(
            [
                [day[feature] for feature in self.REQUIRED_FEATURES]
                for day in days
            ],
            columns=self.REQUIRED_FEATURES,
        )

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        results = []

        for pred, prob in zip(predictions, probabilities):
            label = "good" if pred == 1 else "skip"
            results.append(
                Prediction(
                    label=label,
                    probability=float(prob)
                )
            )

        return results

model_folder = Path(__file__).parent.parent / "models"
model_path = model_folder / "weather_classifier.pkl"
classifier = WeatherClassifier(model_path)

days = [
    {
        "temperature_2m_max": 22,
        "temperature_2m_min": 10,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 12,
    }
]

