

"""
weatherkit package initialization.

Exports the main public classes so callers can import directly from weatherkit.
"""

# As you build modules, add imports here.
# For now, leave placeholders.

try:
    from .records import HourlyReading
except ImportError:
    pass

try:
    from .schemas import HourlyBlock,WeatherResponse
except ImportError:
    pass

try:
    from .summarize import DailyAggregator, DailySummary
except ImportError:
    pass