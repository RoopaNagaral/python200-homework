from dataclasses import dataclass, FrozenInstanceError, field
from pydantic import BaseModel, Field, ValidationError, model_validator
import pytest

# --- Classes ---
# Q1 - Write a class Thermometer that stores a list of temperature readings in Celsius.

class Thermometer:
    def __init__(self, location, readings=[]):
        self.location = location
        self.readings = readings
        
    def add(self, reading):
        self.readings.append(reading)
        
    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)
    
    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)
    
# --- Classes ---
# Q2 - Add a __repr__ to Thermometer that prints something like
    
    def __repr__(self):
        return(
            f"Thermometer(location='Charlotte', n_readings=4, average=18.6)"
        )

my_thermometer = Thermometer("dallas")

print(my_thermometer.add(23))
print(my_thermometer.add(10))
print(my_thermometer.add(3))
print(my_thermometer.add(50))

print("\nAverage of readings:")
print(my_thermometer.average())

print("\nHotest reading:")
print(my_thermometer.hottest())

'''
Subjective Comment: average() must handle empty case because sum(self.readings) / len(self.readings) 
divides by zero when the list is empty, which raises ZeroDivisionError. 
Returning none reports "no data" without crashing.
'''

# Class -Q2
print("\n__repe__ message:")
print(my_thermometer)
print([my_thermometer, my_thermometer])

'''
Subjective Comment: without __repr__. Python prints default value like <__main__.DailyWeather object at 0x104f3a2d0>,
it shows the type and the memory address not the field values. This is unhelpfull when debugging, 
because it list only address not the data.
'''

# --- Classes ---
# Q3 - Write a second class TemperatureAler
class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold
        
    def breaches(self, thermometer):
        mylist = []
        for i in thermometer:
            if i > self.threshold:
                mylist.append(i)
        return mylist            
        
my_threshold1 = TemperatureAlert(15.0)
my_threshold2 = TemperatureAlert(28.0)

print("\nThreshold1:")
print(my_threshold1.breaches(my_thermometer.readings))

print("\nThreshold2:")
print(my_threshold2.breaches(my_thermometer.readings))

'''
Subjective Comment: the threshold lives on TemperatureAlert object so configuration is set once and 
then applied many times. With twenty thermometers to check, the students created alert once and calls breaches() 
on each thermometer, rather than passingthe threshold value through twenty seperate calls.
'''

# --- Dataclass ---
# Q1 - Rewrite this class as a dataclass. Add type hints to every field and a docstring describing what it represents.

@dataclass(frozen=True)
class Station:
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float
    
station_a = Station("11", "Dallas", 12.9, 30.2, 20.45)
station_b = Station("11", "Dallas", 12.9, 30.2, 20.45)

print("\nDataclass Station:")
print(station_a == station_b)

'''
Subjective comment: the result is True because @dataclass generates an __eq__ that compares field values. 
The original hand-written class had no __eq__, so it would compare by object identity and 
print False for two separate instances.
'''

# --- Dataclass ---
# Q2 - Make Station frozen

station_c = Station("15", "Celina", 15.9, 40.2, 28.5)
print("\n")

try:
    """ Modifying the longitude value """
    station_c.longitude = 38.25
except FrozenInstanceError as e:
        """ Returns error while trying to modify the frozen class value """
        print(e)
              
stations = { 
            Station("11", "Dallas", 12.9, 30.2, 20.45),
            Station("11", "Dallas", 12.9, 30.2, 20.45),
            Station("15", "Celina", 15.9, 40.2, 28.5)
}
print("\nLength of stations:")
print(len(stations))

'''
Subjective Comment: besides immutable frozen=True makes dataclass hashable, 
so you can use it as a dictionary key or put it in a set. And it treats two identical fields as the same key, 
which is why the set has two entries and not three.
'''

# --- Dataclass ---
# Q3 - Write a dataclass StationBatch

@dataclass
class StationBatch:
    region: str
    stations: list[Station] = field(default_factory= list)
    
    def add(self, station: list[Station]) -> None:
        """ Adding values to stations list"""
        self.stations.append(station)
        
    def highest(self) -> list[Station] | None:
        """ Returns the station with the greatest elevation, or None if the batch is empty"""
        if not self.stations:
            return None
        return max(self.stations)
    
stations_batch = StationBatch("Dallas")
print("\nStation batch:")
print(stations_batch.add(35.0))
print(stations_batch.add(15.0))
print(stations_batch.add(80.0))

print(stations_batch)

print("\nHighest batch value:")
print(stations_batch.highest())

'''
ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory

The comment explains that a bare default is created once when the class is defined, 
so every instance would share the same list, and dataclasses refuse this to prevent that shared-state bug. 
The fix is field(default_factory=list), which calls list() fresh for each instance.
'''

# --- Pydantic ---
# Q1 - Write a Pydantic model Reading with these fields

class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str 
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)
    
    # --- Pydantic ---
    # Q4 - Add a model_validator(mode="after")
    model_validator(mode="after")
    def reject_failed_sensor(self):
        """ reading where humidity is exactly 0.0 and temperature_c is below -40"""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("humidity 0 with temperature below -40 indicates a failed sensor")
        return self
    
    
my_readings = Reading(station_id="DAL",timestamp= "10:30:05",temperature_c= 31.0, humidity=43.0)
print("\nReadings:")
print(my_readings)

# --- Pydantic ---
# Q2 - Show three separate failures, each wrapped in try / except ValidationError
print("\nErrors:")
try:
    Reading(
        station_id="DAL",
        temperature_c= 150.0, 
        humidity=43.0
    )
    
    Reading(
        station_id="CEL",
        timestamp= "11:12:00",
        temperature_c= 150.0, 
        humidity=43.0
    )
    
    Reading(
        station_id="MCK",
        timestamp= "1:36:05",
        temperature_c= 40.0, 
        humidity="very humid"
    )
    
except ValidationError as e:
    print(e)
 
print("\nReadings passed with different datatype:")       
my_readings1 = Reading(
        station_id="MCK",
        timestamp= "1:36:05",
        temperature_c= "21.5", 
        humidity= 40
)

print(my_readings1)
print(type(my_readings1.temperature_c), type(my_readings1.humidity))


"""
Error 1 : 1 validation error for Reading
timestamp
  Field required [type=missing, input_value={'station_id': 'DAL', 'te... 31.0, 'humidity': 43.0}, input_type=dict]
  
Error 2: 1 validation error for Reading
temperature_c
  Input should be less than or equal to 60 [type=less_than_equal, input_value=150.0, input_type=float]
  
Error 3:1 validation error for Reading
humidity
  Input should be a valid number, unable to parse string as a number [type=float_parsing, input_value='very humid', input_type=str]
"""

"""
Subjective comment: Pydantic accepts "21.5" because it has exactly one reasonable interpretation as a float, 
and it rejects "very humid" because that string has no numeric interpretation. 
The rule is that Pydantic coerces a value when there is a single unambiguous conversion, 
and refuses when the conversion would require guessing.
"""

# --- Pydantic ---
# Q3 - Trigger several errors at once.

print("\nErrors:")
try:
    Reading(
        station_id="TX",
        temperature_c= "10.0", 
        humidity=43.0
    )
    
except ValidationError as e:
    print(f"{e.error_count()} problems found\n")
    for err in e.errors():
        print(f"  field={err['loc']}  type={err['type']}  msg={err['msg']}")
        
"""
Subjective comment: reporting all errors at once is more useful because the student can fix every problem 
in a malformed payload in one pass, instead of fixing one field, re-running, discovering the next error, 
and repeating.
"""

# --- Pydantic ---
# Q4 - Add a model_validator(mode="after")

try:
    my_readings4 = Reading(
        station_id="MCK",
        timestamp= "1:36:05",
        temperature_c= -50.0, 
        humidity=0.0
    )
    print("\n")
    #print(my_readings4.reject_failed_sensor())
except ValidationError as e:
    print(e)
    
"""
Subjective comment: this rule cannot be expressed with Field constraints alone because it compares 
two fields against each other. Field constraints check one field in isolation, and neither 
humidity == 0.0 nor temperature_c < -40 is invalid on its own. Only their combination is invalid, 
and a model_validator(mode="after") is the tool that runs once every field is populated so both can be read together.
"""

# --- pytest ---
# Q1 

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a Celsius temperature to Fahrenheit."""
    return celsius * 9 / 5 + 32

def test_celsius_to_fahrenheit():
    """test converting Celsius temperature to Fahrenheit"""
    assert celsius_to_fahrenheit(0.0) == 32
    assert celsius_to_fahrenheit(100.0) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)
    
"""
Subjective comment: pytest.approx was necessary because floating-point arithmetic does not produce an exact 98.6, 
so a plain == comparison fails on a correct function. pytest.approx compares within a small tolerance.
"""

# --- pytest ---
# Q2

def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list of values")
    return sum(values) / len(values)

def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])
        
"""
Subjective comment: pytest.raises(ValueError) alone would pass on any ValueError, including one accidentally 
raised by a mistake in the test setup. Adding match= confirms the error is the specific one the code is meant to 
raise, so the test cannot pass for the wrong reason.
"""

# --- pytest ---
# Q3

@pytest.mark.parametrize("values",
    [
        ([25,60]),
        ([-30, 4, 80]),
        ([70]),
        ([8, 21, 67, 34, 45]),
    ],
)

def test_mean_values(values):
    assert mean(values)
    
"""
warmup_01.py::test_mean_values[values0] PASSED [ 50%]
warmup_01.py::test_mean_values[values1] PASSED [ 66%]
warmup_01.py::test_mean_values[values2] PASSED [ 83%]
warmup_01.py::test_mean_values[values3] PASSED [100%] 
"""

"""
Subjective comment: one parametrized test with four cases is better than four near-identical test functions because the 
test logic is written once, each case still reports its own pass or fail, and adding a new case is one line rather than a new function.
"""

# --- pytest ---
# Q4

"""
Failure output after deliberately breaking celsius_to_fahrenheit (for example, change 9 / 5 to 9 / 4) 
FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212
"""

"""
Subjective comment: pytest showed the actual computed value against the expected value 
(for example assert 257.0 == 212.0), naming both sides of the failed assertion. 
That is more useful than a bare "assertion failed" because the student can see exactly what the function 
produced and what was expected, which points straight at the bug.
"""