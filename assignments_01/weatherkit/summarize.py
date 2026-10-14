from dataclasses import dataclass
from typing import Dict,List
from collections import defaultdict


from weatherkit.records import HourlyReading 

#Task 4: The Aggregation

@dataclass
class DailySummary:
    """
    A daily weather summary aggregated from hourly readings.

    Attributes:
        date: Calendar date in YYYY-MM-DD format.
        temp_max: Maximum temperature observed that day.
        temp_min: Minimum temperature observed that day.
        precipitation_sum: Total precipitation for the day.
        hours_observed: Number of hourly readings contributing.
    """
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int
    
    def temp_range(self) -> float:
        """Return the difference between max and min temperature."""
        return self.temp_max - self.temp_min
    
    
class DailyAggregator:
    """
    Aggregate hourly readings into daily summaries.

    Parameters:
        min_hours: Minimum number of hourly observations required
                   for a day to be included in the final summary.
    """
    def __init__(self, min_hours: int = 24):
        self.min_hours = min_hours
        
    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """
        Group hourly readings by calendar date and compute daily summaries.

        Days with fewer than `min_hours` observations are excluded.

        Returns:
            A list of DailySummary objects sorted by date.
        """
        grouped: dict[str, list[HourlyReading]] = defaultdict(list)
        
        # Group by date (first 10 chars of timestamp: YYYY-MM-DD)
        for r in readings:
            date = r.timestamp[:10]
            grouped[date].append(r)
            
        summaries: list[DailySummary] = []
        
        for date, items in grouped.items():
            if len(items) < self.min_hours:
                continue
            
            temps = [i.temperature_c for i in items]
            precip = [i.precipitation_mm for i in items]
            
            summary = DailySummary(
                date=date,
                temp_max=max(temps),
                temp_min=min(temps),
                precipitation_sum=sum(precip),
                hours_observed=len(items),
            )
            summaries.append(summary)
            
        # Sort by date string (YYYY-MM-DD sorts lexicographically)
        summaries.sort(key=lambda s: s.date)
        return summaries
    
    def incomplete_days(self, readings: List[HourlyReading]) -> List[str]:
        """
        Return the list of dates that were dropped due to insufficient data.
        """
        grouped: Dict[str, List[HourlyReading]] = defaultdict(list)

        for r in readings:
            date = r.timestamp[:10]
            grouped[date].append(r)

        return sorted(
            [date for date, items in grouped.items() if len(items) < self.min_hours]
        )