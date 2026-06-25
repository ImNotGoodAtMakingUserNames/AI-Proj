"""
Analyze what causes negative duration predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============== FEATURE ENGINEERING ==============

def parseDateTime(dateString):
    """Parse datetime string and return datetime object"""
    if pd.isna(dateString):
        raise ValueError(f"dateString is NaN")
    try:
        s = str(dateString).strip()
        if ',' in s:
            s = s.split(',', 1)[0].strip()
        
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%y %H:%M',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Failed to parse datetime string '{s}' with any supported format")
    except Exception as e:
        raise ValueError(f"Failed to parse datetime string '{dateString}': {e}")

def getStartHour(dateString):
    return int(parseDateTime(dateString).hour)

def getStartMinute(dateString):
    return int(parseDateTime(dateString).minute)

def getStartSecond(dateString):
    return int(parseDateTime(dateString).second)

def getStartDay(dateString):
    return int(parseDateTime(dateString).day)

def getStartMonth(dateString):
    return int(parseDateTime(dateString).month)

def getStartYear(dateString):
    return int(parseDateTime(dateString).year)

def getDayOfWeek(start_month, start_day, start_year):
    if any(pd.isna(x) for x in [start_month, start_day, start_year]):
        raise ValueError(f"One or more datetime components are NaN")
    try:
        dt = datetime(int(start_year), int(start_month), int(start_day))
        return int(dt.weekday())
    except Exception as e:
        raise ValueError(f"Failed to create datetime from components: {e}")

def getSeason(start_month):
    if pd.isna(start_month):
        raise ValueError(f"start_month is NaN")
    month = int(start_month)
    if month in [12, 1, 2]:
        return 4
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    elif month in [9, 10, 11]:
        return 3
    else:
        raise ValueError(f"Invalid month value: {month}")

def isWeekend(day_of_week):
    if day_of_week is None:
        raise ValueError(f"day_of_week is None")
    return int(day_of_week >= 5)

def getTimeofDay(start_hour):
    if start_hour < 8:
        return 0
    elif start_hour < 11:
        return 1
    elif start_hour < 13:
        return 2
    elif start_hour < 16:
        return 3
    elif start_hour < 18:
        return 4
    else:
        return 5

def isHoliday(start_month, start_day, start_year):
    if any(pd.isna(x) for x in [start_month, start_day, start_year]):
        raise ValueError(f"One or more date components are NaN")
    
    month = int(start_month)
    day = int(start_day)
    year = int(start_year)
    
    holidays_2034 = {
        (1, 1), (1, 16), (2, 20), (5, 29), (6, 19),
        (7, 4), (9, 4), (10, 9), (11, 11), (11, 23), (12, 25),
    }
    
    if year != 2034:
        raise ValueError(f"Holiday checking only configured for 2034, got {year}")
    
    return int((month, day) in holidays_2034)

def engineer_features(df):
    """Enhanced feature engineering"""
    taxi = df.copy()
    
    taxi['start_hour'] = taxi['pickup_datetime'].apply(getStartHour)
    taxi['start_minute'] = taxi['pickup_datetime'].apply(getStartMinute)
    taxi['start_second'] = taxi['pickup_datetime'].apply(getStartSecond)
    taxi['start_day'] = taxi['pickup_datetime'].apply(getStartDay)
    taxi['start_month'] = taxi['pickup_datetime'].apply(getStartMonth)
    taxi['start_year'] = taxi['pickup_datetime'].apply(getStartYear)
    
    taxi['dayOfWeek'] = taxi.apply(
        lambda row: getDayOfWeek(row['start_month'], row['start_day'], row['start_year']), axis=1
    )
    
    taxi['x2xDistance'] = abs(taxi['dropoff_x'] - taxi['pickup_x'])
    taxi['y2yDistance'] = abs(taxi['dropoff_y'] - taxi['pickup_y'])
    
    taxi['season'] = taxi['start_month'].apply(getSeason)
    
    taxi['timeOfDay'] = taxi['start_hour'].apply(getTimeofDay)
    taxi['is_weekend'] = taxi['dayOfWeek'].apply(isWeekend)
    taxi['additionalStop'] = (taxi['NumberOfPassengers'] > 1).astype(int)
    taxi['is_holiday'] = taxi.apply(
        lambda row: isHoliday(row['start_month'], row['start_day'], row['start_year']), axis=1
    )
    
    taxi['manhattan_distance'] = taxi['x2xDistance'] + taxi['y2yDistance']
    taxi['euclidean_distance'] = np.sqrt(taxi['x2xDistance'] ** 2 + taxi['y2yDistance'] ** 2)
    
    taxi['manhattan_distance_squared'] = taxi['manhattan_distance'] ** 2
    taxi['euclidean_distance_squared'] = taxi['euclidean_distance'] ** 2
    taxi['euclidean_distance_cubed'] = taxi['euclidean_distance'] ** 3
    
    taxi['distance_x_hour'] = taxi['euclidean_distance'] * taxi['start_hour']
    taxi['distance_x_hour_squared'] = taxi['distance_x_hour'] ** 2
    
    taxi['distance_x_timeOfDay'] = taxi['euclidean_distance'] * taxi['timeOfDay']
    
    taxi['distance_x_weekend'] = taxi['euclidean_distance'] * taxi['is_weekend']
    
    taxi['distance_x_passengers'] = taxi['euclidean_distance'] * taxi['NumberOfPassengers']
    
    taxi['passengers_x_weekend'] = taxi['NumberOfPassengers'] * taxi['is_weekend']
    
    taxi['hour_x_is_weekend'] = taxi['start_hour'] * taxi['is_weekend']
    
    taxi['hour_sin'] = np.sin(2 * np.pi * taxi['start_hour'] / 24)
    taxi['hour_cos'] = np.cos(2 * np.pi * taxi['start_hour'] / 24)
    taxi['month_sin'] = np.sin(2 * np.pi * taxi['start_month'] / 12)
    taxi['month_cos'] = np.cos(2 * np.pi * taxi['start_month'] / 12)
    taxi['dayOfWeek_sin'] = np.sin(2 * np.pi * taxi['dayOfWeek'] / 7)
    taxi['dayOfWeek_cos'] = np.cos(2 * np.pi * taxi['dayOfWeek'] / 7)
    
    taxi['is_rush_hour'] = (
        ((taxi['start_hour'] >= 8) & (taxi['start_hour'] <= 10)) |
        ((taxi['start_hour'] >= 17) & (taxi['start_hour'] <= 19))
    ).astype(int)
    
    taxi['is_night'] = ((taxi['start_hour'] >= 22) | (taxi['start_hour'] < 5)).astype(int)
    
    taxi['distance_category'] = pd.cut(
        taxi['euclidean_distance'], 
        bins=[0, 0.5, 1, 2, 5, 10, float('inf')], 
        labels=['very_short', 'short', 'medium', 'long', 'very_long', 'very_long_plus']
    ).cat.codes
    
    taxi['solo_passenger'] = (taxi['NumberOfPassengers'] == 1).astype(int)
    taxi['many_passengers'] = (taxi['NumberOfPassengers'] > 3).astype(int)
    
    taxi['minutes_since_midnight'] = taxi['start_hour'] * 60 + taxi['start_minute']
    taxi['minutes_until_midnight'] = 24 * 60 - taxi['minutes_since_midnight']
    
    taxi['pickup_quadrant_x'] = (taxi['pickup_x'] > taxi['pickup_x'].median()).astype(int)
    taxi['pickup_quadrant_y'] = (taxi['pickup_y'] > taxi['pickup_y'].median()).astype(int)
    taxi['dropoff_quadrant_x'] = (taxi['dropoff_x'] > taxi['dropoff_x'].median()).astype(int)
    taxi['dropoff_quadrant_y'] = (taxi['dropoff_y'] > taxi['dropoff_y'].median()).astype(int)
    
    taxi['cross_quadrant_trip'] = (
        (taxi['pickup_quadrant_x'] != taxi['dropoff_quadrant_x']) |
        (taxi['pickup_quadrant_y'] != taxi['dropoff_quadrant_y'])
    ).astype(int)
    
    return taxi

# ============== MAIN ANALYSIS ==============

print("="*80)
print("ANALYZING NEGATIVE PREDICTIONS")
print("="*80)

# Load test predictions
print("\nLoading test data with predictions...")
test_predictions = pd.read_csv('./test_files/Gotham_Test_Set_with_predictions.csv')
print(f"✓ Loaded {len(test_predictions):,} test rows")

# Load original test data to re-engineer features
print("Loading original test data...")
test_original = pd.read_csv('./test_files/Gotham_Test_Set.csv')

# Engineer features
print("Engineering features...")
test_features = engineer_features(test_original)

# Identify negative predictions
negative_mask = test_predictions['duration'] < 0
positive_mask = test_predictions['duration'] >= 0
negative_indices = np.where(negative_mask)[0]

print(f"\n{'='*80}")
print(f"NEGATIVE PREDICTIONS OVERVIEW")
print(f"{'='*80}")
print(f"Total negative predictions: {negative_mask.sum()}")
print(f"Total positive predictions: {positive_mask.sum()}")
print(f"Percentage negative: {(negative_mask.sum() / len(test_predictions) * 100):.2f}%")

# Compare feature statistics between negative and positive predictions
print(f"\n{'='*80}")
print(f"FEATURE COMPARISON: NEGATIVE vs POSITIVE PREDICTIONS")
print(f"{'='*80}")

key_features = [
    'euclidean_distance', 'manhattan_distance', 
    'start_hour', 'NumberOfPassengers', 'timeOfDay',
    'is_weekend', 'is_rush_hour', 'distance_x_hour',
    'distance_x_passengers', 'hour_x_is_weekend'
]

print(f"\n{'Feature':<25} {'Negative Mean':<20} {'Positive Mean':<20} {'Difference':<15}")
print(f"{'-'*80}")

for feature in key_features:
    if feature in test_features.columns:
        neg_mean = test_features.loc[negative_mask, feature].mean()
        pos_mean = test_features.loc[positive_mask, feature].mean()
        diff = neg_mean - pos_mean
        
        print(f"{feature:<25} {neg_mean:<20.4f} {pos_mean:<20.4f} {diff:+.4f}")

# Analyze specific patterns
print(f"\n{'='*80}")
print(f"PATTERN ANALYSIS: CONDITIONS IN NEGATIVE PREDICTIONS")
print(f"{'='*80}")

# Distance analysis
short_trips_negative = (test_features.loc[negative_mask, 'euclidean_distance'] < 0.5).sum()
short_trips_positive = (test_features.loc[positive_mask, 'euclidean_distance'] < 0.5).sum()

print(f"\nShort trips (distance < 0.5):")
print(f"  Negative predictions: {short_trips_negative} ({short_trips_negative/negative_mask.sum()*100:.1f}%)")
print(f"  Positive predictions: {short_trips_positive} ({short_trips_positive/positive_mask.sum()*100:.1f}%)")

# Night time analysis
night_negative = test_features.loc[negative_mask, 'is_night'].sum()
night_positive = test_features.loc[positive_mask, 'is_night'].sum()

print(f"\nNight time trips (22:00 - 04:59):")
print(f"  Negative predictions: {night_negative} ({night_negative/negative_mask.sum()*100:.1f}%)")
print(f"  Positive predictions: {night_positive} ({night_positive/positive_mask.sum()*100:.1f}%)")

# Weekend analysis
weekend_negative = test_features.loc[negative_mask, 'is_weekend'].sum()
weekend_positive = test_features.loc[positive_mask, 'is_weekend'].sum()

print(f"\nWeekend trips:")
print(f"  Negative predictions: {weekend_negative} ({weekend_negative/negative_mask.sum()*100:.1f}%)")
print(f"  Positive predictions: {weekend_positive} ({weekend_positive/positive_mask.sum()*100:.1f}%)")

# Solo passenger analysis
solo_negative = test_features.loc[negative_mask, 'solo_passenger'].sum()
solo_positive = test_features.loc[positive_mask, 'solo_passenger'].sum()

print(f"\nSolo passenger trips:")
print(f"  Negative predictions: {solo_negative} ({solo_negative/negative_mask.sum()*100:.1f}%)")
print(f"  Positive predictions: {solo_positive} ({solo_positive/positive_mask.sum()*100:.1f}%)")

# Show top 20 rows with most negative predictions
print(f"\n{'='*80}")
print(f"TOP 20 MOST NEGATIVE PREDICTIONS - FEATURE ANALYSIS")
print(f"{'='*80}")

# Sort by prediction value
sorted_indices = np.argsort(test_predictions['duration'])
most_negative_indices = sorted_indices[:20]

print(f"\n{'Rank':<6} {'Pred (s)':<12} {'Distance':<12} {'Hour':<6} {'Pass':<6} {'Night':<6} {'RushHr':<6}")
print(f"{'-'*80}")

for rank, idx in enumerate(most_negative_indices, 1):
    pred = test_predictions.iloc[idx]['duration']
    dist = test_features.iloc[idx]['euclidean_distance']
    hour = test_features.iloc[idx]['start_hour']
    passengers = test_features.iloc[idx]['NumberOfPassengers']
    is_night = test_features.iloc[idx]['is_night']
    is_rush = test_features.iloc[idx]['is_rush_hour']
    
    print(f"{rank:<6} {pred:<12.2f} {dist:<12.4f} {hour:<6.0f} {passengers:<6.0f} {is_night:<6.0f} {is_rush:<6.0f}")

# Correlation analysis with distance
print(f"\n{'='*80}")
print(f"DISTANCE PATTERNS IN NEGATIVE PREDICTIONS")
print(f"{'='*80}")

distance_bins = [0, 0.1, 0.5, 1, 2, 5, 100]
distance_labels = ['0-0.1', '0.1-0.5', '0.5-1', '1-2', '2-5', '5+']

test_features['distance_bin'] = pd.cut(test_features['euclidean_distance'], bins=distance_bins, labels=distance_labels)

print(f"\n{'Distance Range':<20} {'Negative %':<15} {'Count Negative':<15} {'Count Total':<15}")
print(f"{'-'*80}")

for label in distance_labels:
    bin_mask = test_features['distance_bin'] == label
    bin_negative = (negative_mask & bin_mask).sum()
    bin_total = bin_mask.sum()
    if bin_total > 0:
        pct = bin_negative / bin_total * 100
        print(f"{label:<20} {pct:<15.1f} {bin_negative:<15} {bin_total:<15}")

print(f"\n{'='*80}")
print("Analysis complete!")
print(f"{'='*80}")
