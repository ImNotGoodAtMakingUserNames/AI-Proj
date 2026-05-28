# Imports
from enum import IntEnum
from dataclasses import dataclass
import pandas as pd
from datetime import date, datetime, timedelta


# 2034-01-30 10:24:44,1,161.6215557,378.3926802,154.5766309,357.1002292
def parseDateTime(dateString):
    """Parse datetime string and return datetime object, raise exception on error"""
    if pd.isna(dateString):
        raise ValueError(f"dateString is NaN")
    try:
        s = str(dateString).strip()
        if ',' in s:
            s = s.split(',', 1)[0].strip()
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        raise ValueError(f"Failed to parse datetime string '{dateString}': {e}")

# Granular datetime component extractors
def getStartHour(dateString):
    """Extract hour from pickup_datetime"""
    dt = parseDateTime(dateString)
    return int(dt.hour)

def getStartMinute(dateString):
    """Extract minute from pickup_datetime"""
    dt = parseDateTime(dateString)
    return int(dt.minute)

def getStartSecond(dateString):
    """Extract second from pickup_datetime"""
    dt = parseDateTime(dateString)
    return int(dt.second)

def getStartDay(dateString):
    """Extract day from pickup_datetime"""
    dt = parseDateTime(dateString)
    return int(dt.day)

def getStartMonth(dateString):
    """Extract month from pickup_datetime"""
    dt = parseDateTime(dateString)
    return int(dt.month)

def getStartYear(dateString):
    """Extract year from pickup_datetime"""
    dt = parseDateTime(dateString)
    return int(dt.year)

def getDayOfWeek(start_month, start_day, start_year):
    """Get day of week from individual datetime components"""
    if any(pd.isna(x) for x in [start_month, start_day, start_year]):
        raise ValueError(f"One or more datetime components are NaN: month={start_month}, day={start_day}, year={start_year}")
    try:
        dt = datetime(int(start_year), int(start_month), int(start_day))
        return int(dt.weekday())
    except Exception as e:
        raise ValueError(f"Failed to create datetime from components - month={start_month}, day={start_day}, year={start_year}: {e}")

def getSeason(start_month):
    """Get season from month component"""
    if pd.isna(start_month):
        raise ValueError(f"start_month is NaN")
    month = int(start_month)
    if month in [12, 1, 2]:
        return 4  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    elif month in [9, 10, 11]:
        return 3  # Fall
    else:
        raise ValueError(f"Invalid month value: {month}. Must be 1-12.")

def isWeekend(day_of_week):
    """Check if the day is a weekend (Saturday or Sunday)"""
    if day_of_week is None:
        raise ValueError(f"day_of_week is None")
    # weekday() returns 5 for Saturday, 6 for Sunday
    return day_of_week >= 5

def getEndTime(start_hour, start_minute, start_second, start_day, start_month, start_year, duration):
    """Calculate end_time by adding duration (in seconds) to pickup_datetime"""
    if any(pd.isna(x) for x in [start_hour, start_minute, start_second, start_day, start_month, start_year, duration]):
        raise ValueError(f"One or more datetime components are NaN: hour={start_hour}, minute={start_minute}, second={start_second}, day={start_day}, month={start_month}, year={start_year}, duration={duration}")
    try:
        dt = datetime(int(start_year), int(start_month), int(start_day),
                      int(start_hour), int(start_minute), int(start_second))
        end_time = dt + timedelta(seconds=int(duration))
        return end_time
    except Exception as e:
        raise ValueError(f"Failed to create end_time from components - hour={start_hour}, minute={start_minute}, second={start_second}, day={start_day}, month={start_month}, year={start_year}, duration={duration}: {e}")


# Read Data
df = pd.read_csv('Train.csv')
taxi = df.copy()

# Extract granular datetime components from pickup_datetime
taxi['start_hour'] = taxi['pickup_datetime'].apply(getStartHour)
# print(taxi['start_hour'])
taxi['start_minute'] = taxi['pickup_datetime'].apply(getStartMinute)
taxi['start_second'] = taxi['pickup_datetime'].apply(getStartSecond)
taxi['start_day'] = taxi['pickup_datetime'].apply(getStartDay)
taxi['start_month'] = taxi['pickup_datetime'].apply(getStartMonth)
taxi['start_year'] = taxi['pickup_datetime'].apply(getStartYear)

# Derive features from granular components
taxi['dayOfWeek'] = taxi.apply(lambda row: getDayOfWeek(row['start_month'], row['start_day'], row['start_year']), axis=1)
# print(taxi['dayOfWeek'])

taxi['x2xDistance'] = abs(taxi['dropoff_x'] - taxi['pickup_x'])
# print(taxi['x2xDistance'])

taxi['y2yDistance'] = abs(taxi['dropoff_y'] - taxi['pickup_y'])

taxi['season'] = taxi['start_month'].apply(getSeason)
# print(taxi['season'])

taxi['end_time'] = taxi.apply(lambda row: getEndTime(row['start_hour'], row['start_minute'], row['start_second'], row['start_day'], row['start_month'], row['start_year'], row['duration']), axis=1)
# print(taxi['end_time'])

taxi['is_weekend'] = taxi['dayOfWeek'].apply(isWeekend)
# print(taxi['is_weekend'])

taxi['additionalStop'] = taxi['NumberOfPassengers'] > 1
# print(taxi['additionalStop'])