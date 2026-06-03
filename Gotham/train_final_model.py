"""
Final XGBoost Model Training - Full Dataset
Uses 20 selected features and tuned hyperparameters from Pass 2
Ready for test data prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============== FEATURE ENGINEERING ==============

def parseDateTime(dateString):
    """Parse datetime string and return datetime object"""
    if pd.isna(dateString):
        raise ValueError(f"dateString is NaN")
    try:
        s = str(dateString).strip()
        if ',' in s:
            s = s.split(',', 1)[0].strip()
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
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

def getEndTime(start_hour, start_minute, start_second, start_day, start_month, start_year, duration):
    if any(pd.isna(x) for x in [start_hour, start_minute, start_second, start_day, start_month, start_year, duration]):
        raise ValueError(f"One or more datetime components are NaN")
    try:
        dt = datetime(int(start_year), int(start_month), int(start_day),
                      int(start_hour), int(start_minute), int(start_second))
        end_time = dt + timedelta(seconds=int(duration))
        return end_time
    except Exception as e:
        raise ValueError(f"Failed to create end_time: {e}")

def getEndHour(end_time):
    if pd.isna(end_time):
        raise ValueError("end_time is NaN")
    return int(end_time.hour)

def getEndMinute(end_time):
    if pd.isna(end_time):
        raise ValueError("end_time is NaN")
    return int(end_time.minute)

def getEndSecond(end_time):
    if pd.isna(end_time):
        raise ValueError("end_time is NaN")
    return int(end_time.second)

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
    
    # Extract datetime components
    taxi['start_hour'] = taxi['pickup_datetime'].apply(getStartHour)
    taxi['start_minute'] = taxi['pickup_datetime'].apply(getStartMinute)
    taxi['start_second'] = taxi['pickup_datetime'].apply(getStartSecond)
    taxi['start_day'] = taxi['pickup_datetime'].apply(getStartDay)
    taxi['start_month'] = taxi['pickup_datetime'].apply(getStartMonth)
    taxi['start_year'] = taxi['pickup_datetime'].apply(getStartYear)
    
    # Derived features
    taxi['dayOfWeek'] = taxi.apply(
        lambda row: getDayOfWeek(row['start_month'], row['start_day'], row['start_year']), axis=1
    )
    
    taxi['x2xDistance'] = abs(taxi['dropoff_x'] - taxi['pickup_x'])
    taxi['y2yDistance'] = abs(taxi['dropoff_y'] - taxi['pickup_y'])
    
    taxi['season'] = taxi['start_month'].apply(getSeason)
    
    taxi['end_time'] = taxi.apply(
        lambda row: getEndTime(row['start_hour'], row['start_minute'], row['start_second'],
                              row['start_day'], row['start_month'], row['start_year'], row['duration']), axis=1
    )
    
    taxi['end_hour'] = taxi['end_time'].apply(getEndHour)
    taxi['end_minute'] = taxi['end_time'].apply(getEndMinute)
    taxi['end_second'] = taxi['end_time'].apply(getEndSecond)
    
    taxi['timeOfDay'] = taxi['start_hour'].apply(getTimeofDay)
    taxi['is_weekend'] = taxi['dayOfWeek'].apply(isWeekend)
    taxi['additionalStop'] = (taxi['NumberOfPassengers'] > 1).astype(int)
    taxi['is_holiday'] = taxi.apply(
        lambda row: isHoliday(row['start_month'], row['start_day'], row['start_year']), axis=1
    )
    
    # Distance metrics
    taxi['manhattan_distance'] = taxi['x2xDistance'] + taxi['y2yDistance']
    taxi['euclidean_distance'] = np.sqrt(taxi['x2xDistance'] ** 2 + taxi['y2yDistance'] ** 2)
    
    # Polynomial distance features
    taxi['manhattan_distance_squared'] = taxi['manhattan_distance'] ** 2
    taxi['euclidean_distance_squared'] = taxi['euclidean_distance'] ** 2
    taxi['euclidean_distance_cubed'] = taxi['euclidean_distance'] ** 3
    
    # Interaction: Distance × Hour
    taxi['distance_x_hour'] = taxi['euclidean_distance'] * taxi['start_hour']
    taxi['distance_x_hour_squared'] = taxi['distance_x_hour'] ** 2
    
    # Interaction: Distance × Time of Day
    taxi['distance_x_timeOfDay'] = taxi['euclidean_distance'] * taxi['timeOfDay']
    
    # Interaction: Distance × Weekend
    taxi['distance_x_weekend'] = taxi['euclidean_distance'] * taxi['is_weekend']
    
    # Interaction: Hour × Is_Weekend
    taxi['hour_x_is_weekend'] = taxi['start_hour'] * taxi['is_weekend']
    
    # Interaction: Distance × Passengers
    taxi['distance_x_passengers'] = taxi['euclidean_distance'] * taxi['NumberOfPassengers']
    
    # Cyclic encodings
    taxi['hour_sin'] = np.sin(2 * np.pi * taxi['start_hour'] / 24)
    taxi['hour_cos'] = np.cos(2 * np.pi * taxi['start_hour'] / 24)
    taxi['month_sin'] = np.sin(2 * np.pi * taxi['start_month'] / 12)
    taxi['month_cos'] = np.cos(2 * np.pi * taxi['start_month'] / 12)
    taxi['dayOfWeek_sin'] = np.sin(2 * np.pi * taxi['dayOfWeek'] / 7)
    taxi['dayOfWeek_cos'] = np.cos(2 * np.pi * taxi['dayOfWeek'] / 7)
    
    # Rush hour indicator
    taxi['is_rush_hour'] = (
        ((taxi['start_hour'] >= 8) & (taxi['start_hour'] <= 10)) |
        ((taxi['start_hour'] >= 17) & (taxi['start_hour'] <= 19))
    ).astype(int)
    
    # Night time
    taxi['is_night'] = ((taxi['start_hour'] >= 22) | (taxi['start_hour'] < 5)).astype(int)
    
    # Distance categories
    taxi['distance_category'] = pd.cut(
        taxi['euclidean_distance'], 
        bins=[0, 0.5, 1, 2, 5, 10, float('inf')], 
        labels=['very_short', 'short', 'medium', 'long', 'very_long', 'very_long_plus']
    ).cat.codes
    
    # Passenger category
    taxi['solo_passenger'] = (taxi['NumberOfPassengers'] == 1).astype(int)
    taxi['many_passengers'] = (taxi['NumberOfPassengers'] > 3).astype(int)
    
    # Time-based
    taxi['minutes_since_midnight'] = taxi['start_hour'] * 60 + taxi['start_minute']
    taxi['minutes_until_midnight'] = 24 * 60 - taxi['minutes_since_midnight']
    
    # Interaction: Passengers × Is_Weekend
    taxi['passengers_x_weekend'] = taxi['NumberOfPassengers'] * taxi['is_weekend']
    
    # Geospatial
    taxi['pickup_quadrant_x'] = (taxi['pickup_x'] > taxi['pickup_x'].median()).astype(int)
    taxi['pickup_quadrant_y'] = (taxi['pickup_y'] > taxi['pickup_y'].median()).astype(int)
    taxi['dropoff_quadrant_x'] = (taxi['dropoff_x'] > taxi['dropoff_x'].median()).astype(int)
    taxi['dropoff_quadrant_y'] = (taxi['dropoff_y'] > taxi['dropoff_y'].median()).astype(int)
    
    taxi['cross_quadrant_trip'] = (
        (taxi['pickup_quadrant_x'] != taxi['dropoff_quadrant_x']) |
        (taxi['pickup_quadrant_y'] != taxi['dropoff_quadrant_y'])
    ).astype(int)
    
    return taxi

# ============== MODEL TRAINING ==============

if __name__ == "__main__":
    
    print("="*70)
    print("XGBOOST FINAL TRAINING - FULL DATASET")
    print("="*70)
    
    # Load FULL training data
    print("\nLoading full training dataset...")
    df = pd.read_csv('Train.csv')
    print(f"Total samples: {len(df):,}")
    
    # Feature engineering
    print("Applying feature engineering...")
    taxi = engineer_features(df)
    
    # Selected features from Pass 2 (20 features)
    selected_features = [
        'euclidean_distance', 'manhattan_distance', 'euclidean_distance_squared',
        'hour_cos', 'dayOfWeek', 'end_hour', 'cross_quadrant_trip', 'dayOfWeek_cos',
        'end_minute', 'minutes_until_midnight', 'start_minute', 'distance_category',
        'distance_x_hour', 'dayOfWeek_sin', 'minutes_since_midnight', 'dropoff_y',
        'start_month', 'hour_sin', 'x2xDistance', 'distance_x_timeOfDay'
    ]
    
    X = taxi[selected_features]
    y = taxi['duration']
    
    print(f"\nFeatures: {len(selected_features)}")
    print(f"Samples: {len(X):,}")
    print(f"Target stats:")
    print(f"  Mean: {y.mean():.2f}s")
    print(f"  Median: {y.median():.2f}s")
    print(f"  Std: {y.std():.2f}s")
    print(f"  Min: {y.min():.2f}s")
    print(f"  Max: {y.max():.2f}s")
    
    # Best hyperparameters from Pass 2
    best_params = {
        'n_estimators': 200,  # From best hyperparameters
        'max_depth': 9,
        'learning_rate': 0.15,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'gamma': 0.5,
        'reg_alpha': 0.01,
        'reg_lambda': 2,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'hist'
    }
    
    print(f"\nBest Hyperparameters (from Pass 2):")
    for param, value in best_params.items():
        if param not in ['random_state', 'n_jobs', 'verbosity', 'tree_method']:
            print(f"  {param}: {value}")
    
    # Verify with 5-fold CV on full dataset
    print(f"\n{'='*70}")
    print("VERIFYING WITH 5-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    
    model_cv = xgb.XGBRegressor(**best_params)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_cv, X, y, cv=cv, scoring='r2', n_jobs=-1)
    
    print(f"\nCV R² Scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.6f}")
    print(f"Std CV R²: {cv_scores.std():.6f}")
    
    # Train final model on full dataset
    print(f"\n{'='*70}")
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print(f"{'='*70}")
    
    final_model = xgb.XGBRegressor(**best_params)
    print("\nFitting model on full dataset...")
    final_model.fit(X, y, verbose=False)
    
    # Training set performance
    train_r2 = final_model.score(X, y)
    train_pred = final_model.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, train_pred))
    train_mae = mean_absolute_error(y, train_pred)
    
    print(f"\nTraining Set Performance:")
    print(f"  R²: {train_r2:.6f}")
    print(f"  RMSE: {train_rmse:.2f}s")
    print(f"  MAE: {train_mae:.2f}s")
    
    # Save model
    print(f"\n{'='*70}")
    print("SAVING MODEL")
    print(f"{'='*70}")
    
    model_path = './results_logs/final_xgboost_model.pkl'
    Path('./results_logs').mkdir(exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Model saved to: {model_path}")
    
    # Save feature list for reference
    features_path = './results_logs/selected_features.txt'
    with open(features_path, 'w') as f:
        f.write('\n'.join(selected_features))
    print(f"Feature list saved to: {features_path}")
    
    # Save hyperparameters
    import json
    params_path = './results_logs/final_hyperparameters.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Hyperparameters saved to: {params_path}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nModel trained and ready for test predictions!")
    print(f"CV R² on full dataset: {cv_scores.mean():.6f}")
    print(f"Features used: {len(selected_features)}")
    print(f"Samples used: {len(X):,}")
    print(f"\nTo make predictions on test data:")
    print(f"  1. Load model: pickle.load(open('{model_path}', 'rb'))")
    print(f"  2. Engineer test features using same functions")
    print(f"  3. Select only the {len(selected_features)} features listed in {features_path}")
    print(f"  4. Call model.predict(X_test)")
    print(f"\n{'='*70}")
