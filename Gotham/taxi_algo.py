# Imports
from enum import IntEnum
from dataclasses import dataclass

class date:
    year: int
    month: int
    day: int

class time:
    hour: int
    minute: int
    second: int

class position:
    x: float
    y: float

class holiday:
    New_Years: date
    MLKJ: date
    Inaguration: date
    Presidents_Day: date
    Memorial: date
    Juneteenth: date
    Labor: date
    Columbus: date
    Veterans: date
    Thanksgiving: date
    Christmas: date

class trip:
    pickup_date: date
    pickup_time: time
    pickup_pos: position
    dropoff_pos: position
    distance_x_x: float
    distance_x_y: float
    distance_y_y: float
    distance_y_x: float

    passenger_count: int
    duration: float
    day_of_week: DayOfWeek
    is_holiday: bool


class DayOfWeek(IntEnum):
    SUNDAY = 0
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6


# def calcDayOfWeek(day):

#     if(day !< 0 & day !> 6){

#     }
#     else{
        
#     }


# Read Data




# Calculate day of week function


# Get the day of the week
# 2034-01-01 is a Sunday


# 2034-01-30