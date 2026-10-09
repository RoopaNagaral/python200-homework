import pytest
from weatherkit.summarize import DailySummary, DailyAggregator
from weatherkit.records import HourlyReading

@pytest.fixture
def sample_readings() -> list[HourlyReading]:
    # Two dates: 2026-04-08 and 2026-04-09
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.0),
        HourlyReading("2026-04-08T01:00", 12.0, 0.1),
        HourlyReading("2026-04-09T00:00", 5.0, 0.2),
        HourlyReading("2026-04-09T01:00", 7.0, 0.3),
    ]


def test_grouping_produces_two_summaries(sample_readings):
    agg = DailyAggregator(min_hours=1)
    summaries = agg.summarize(sample_readings)
    dates = [s.date for s in summaries]
    assert sorted(dates) == ["2026-04-08", "2026-04-09"]
    
@pytest.mark.parametrize(
    "date, expected_max, expected_min",
    [
        ("2026-04-08", 12.0, 10.0),
        ("2026-04-09", 7.0, 5.0),
    ],
)
def test_temp_max_and_min_are_correct(sample_readings, date, expected_max, expected_min):
    agg = DailyAggregator(min_hours=1)
    summaries = agg.summarize(sample_readings)
    summary = next(s for s in summaries if s.date == date)
    assert summary.temp_max == expected_max
    assert summary.temp_min == expected_min
    
def test_precipitation_sum_adds_up_correctly(sample_readings):
    agg = DailyAggregator(min_hours=1)
    summaries = agg.summarize(sample_readings)

    s_0808 = next(s for s in summaries if s.date == "2026-04-08")
    s_0809 = next(s for s in summaries if s.date == "2026-04-09")

    assert s_0808.precipitation_sum == pytest.approx(0.1)
    assert s_0809.precipitation_sum == pytest.approx(0.5)


def test_day_with_fewer_than_min_hours_is_dropped_and_reported():
    readings = [
        HourlyReading("2026-04-08T00:00", 10.0, 0.0),
        HourlyReading("2026-04-08T01:00", 12.0, 0.1),
    ]

    agg = DailyAggregator(min_hours=3)
    summaries = agg.summarize(readings)
    incomplete = agg.incomplete_days(readings)

    assert summaries == []  # dropped
    assert "2026-04-08" in incomplete


def test_lowering_min_hours_keeps_previous_incomplete_day():
    readings = [
        HourlyReading("2026-04-08T00:00", 10.0, 0.0),
        HourlyReading("2026-04-08T01:00", 12.0, 0.1),
    ]

    agg = DailyAggregator(min_hours=2)
    summaries = agg.summarize(readings)
    dates = [s.date for s in summaries]

    assert "2026-04-08" in dates
    assert summaries[0].hours_observed == 2