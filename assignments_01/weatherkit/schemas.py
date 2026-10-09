from pydantic import BaseModel, Field, ValidationError, model_validator


#Task 2: The Boundary 
class HourlyBlock(BaseModel):
    """The columnar hourly arrays from an Open-Meteo response."""
    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]
    
    @model_validator(mode="after")
    def lists_same_length(self):
        """All three parallel lists must be the same length."""
        if not (len(self.time) == len(self.temperature_2m) == len(self.precipitation)):
            raise ValueError("hourly lists must all be the same length")
        return self
    
    
class WeatherResponse(BaseModel):
    """A validated hourly Open-Meteo weather response."""
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock
    

