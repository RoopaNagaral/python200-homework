import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

from weather_model import WeatherClassifier, Prediction

# Task 4: The Test Suite
@pytest.fixture
def classifier(tmp_path):
    """Create and save a tiny model, then load WeatherClassifier."""

    X = pd.DataFrame(
        [
            [22, 10, 0, 10],    # good
            [20, 8, 1, 12],     # good
            [-5, -10, 15, 45],  # skip
            [0, -3, 10, 35],    # skip
        ],
        columns=[
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
    )

    y = [1, 1, 0, 0]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )

    model.fit(X, y)

    model_path = tmp_path / "weather_classifier.pkl"
    joblib.dump(model, model_path)

    return WeatherClassifier(str(model_path))


@pytest.mark.parametrize(
    "day,expected_label",
    [
        (
            {
                "temperature_2m_max": 24,
                "temperature_2m_min": 12,
                "precipitation_sum": 0,
                "wind_speed_10m_max": 8,
            },
            "good",
        ),
        (
            {
                "temperature_2m_max": -2,
                "temperature_2m_min": -8,
                "precipitation_sum": 20,
                "wind_speed_10m_max": 40,
            },
            "skip",
        ),
    ],
)
def test_predict_expected_labels(classifier, day, expected_label):
    """Checks both good and bad day classifications."""
    result = classifier.predict([day])[0]

    assert result.label == expected_label


def test_probability_between_zero_and_one(classifier):
    day = {
        "temperature_2m_max": 22,
        "temperature_2m_min": 10,
        "precipitation_sum": 0,
        "wind_speed_10m_max": 10,
    }

    result = classifier.predict([day])[0]

    assert 0.0 <= result.probability <= 1.0


def test_three_days_returns_three_predictions_in_order(classifier):
    days = [
        {
            "temperature_2m_max": 24,
            "temperature_2m_min": 12,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 8,
        },
        {
            "temperature_2m_max": -5,
            "temperature_2m_min": -10,
            "precipitation_sum": 12,
            "wind_speed_10m_max": 40,
        },
        {
            "temperature_2m_max": 21,
            "temperature_2m_min": 9,
            "precipitation_sum": 1,
            "wind_speed_10m_max": 12,
        },
    ]

    results = classifier.predict(days)

    assert len(results) == 3
    assert all(isinstance(r, Prediction) for r in results)

    # Verify predictions are returned in input order.
    assert results[0].label == "good"
    assert results[1].label == "skip"


def test_empty_input_returns_empty_list(classifier):
    assert classifier.predict([]) == []


def test_missing_feature_raises_value_error(classifier):
    bad_day = {
        "temperature_2m_max": 22,
        "temperature_2m_min": 10,
        "precipitation_sum": 0,
        # wind_speed_10m_max missing
    }

    with pytest.raises(ValueError, match="wind_speed_10m_max"):
        classifier.predict([bad_day])


def test_missing_model_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        WeatherClassifier("does_not_exist.pkl")


# Manual test verification:
# I temporarily changed classifier.py to always return "skip":
#
#     label = "skip"
#
# The test `test_predict_expected_labels` failed because the
# clearly good day was expected to be labeled "good".
# After confirming the failure, I restored the original code.