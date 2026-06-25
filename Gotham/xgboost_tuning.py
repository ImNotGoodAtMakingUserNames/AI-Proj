"""
XGBoost Hyperparameter Tuning
Performs RandomizedSearchCV to find optimal hyperparameters
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json
import os
import time
from pathlib import Path

from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import matplotlib.pyplot as plt

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

# ============== XGBOOST TUNING ==============

def tune_xgboost(X, y, feature_names, cv_splits=5, n_iter=80, random_state=42):
    """
    Perform RandomizedSearchCV for XGBoost hyperparameter tuning
    """
    
    # Parameter space for tuning
    param_dist = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [4, 5, 6, 7, 8, 9, 10, 11],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.5, 1, 2],
    }
    
    print(f"\n{'='*70}")
    print(f"XGBOOST HYPERPARAMETER TUNING")
    print(f"{'='*70}")
    print(f"Features: {len(feature_names)} | Samples: {len(X):,}")
    print(f"Testing {n_iter} parameter combinations with {cv_splits}-fold CV")
    print(f"Total fits: {n_iter * cv_splits}\n")
    
    start_time = time.time()
    
    base_model = xgb.XGBRegressor(
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        tree_method='hist'
    )
    
    random_search = RandomizedSearchCV(
        base_model,
        param_dist,
        n_iter=n_iter,
        cv=cv_splits,
        scoring='r2',
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    print(f"Starting fit at {datetime.now().strftime('%H:%M:%S')}...")
    random_search.fit(X, y)
    
    elapsed = time.time() - start_time
    print(f"Fit completed in {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    
    # Results
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_
    
    print(f"\nTuning Results:")
    print(f"  Best CV R² Score: {best_cv_score:.6f}")
    print(f"\n  Best Hyperparameters:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    
    # Top 5 results
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    print(f"\n  Top 5 Configurations:")
    print(results_df[['rank_test_score', 'mean_test_score', 'std_test_score']].head(5).to_string())
    
    return {
        'model': best_model,
        'best_params': best_params,
        'best_cv_score': best_cv_score,
        'search_results': results_df,
        'full_results': random_search,
        'elapsed_minutes': elapsed / 60
    }

def analyze_feature_importance(model, feature_names, importance_threshold=0.01):
    """Analyze and display feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None, None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*70}")
    print(importance_df.head(25).to_string(index=False))
    
    # Identify high-importance features
    high_importance = importance_df[importance_df['Importance'] >= importance_threshold]
    low_importance = importance_df[importance_df['Importance'] < importance_threshold]
    
    print(f"\nFeatures with importance >= {importance_threshold}:")
    print(f"  Count: {len(high_importance)}")
    print(f"  Total importance: {high_importance['Importance'].sum():.4f}")
    
    print(f"\nFeatures with importance < {importance_threshold}:")
    print(f"  Count: {len(low_importance)}")
    print(f"  Total importance: {low_importance['Importance'].sum():.4f}")
    
    return importance_df, high_importance['Feature'].tolist()

def save_tuning_results(model_name, tuning_results, importance_df, log_dir='./results_logs'):
    """Save tuning results"""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save best parameters
    params_path = os.path.join(log_dir, f'xgboost_best_params_{timestamp}.json')
    with open(params_path, 'w') as f:
        json.dump(tuning_results['best_params'], f, indent=2)
    print(f"Saved best parameters to: {params_path}")
    
    # Save feature importance
    if importance_df is not None:
        importance_path = os.path.join(log_dir, f'feature_importance_{timestamp}.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"Saved feature importance to: {importance_path}")
    
    return params_path

# ============== MAIN EXECUTION ==============

if __name__ == "__main__":
    
    print("="*70)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('Train.csv')
    
    # Validate 'duration' column exists and is valid
    print("\nValidating 'duration' target variable...")
    if 'duration' not in df.columns:
        raise ValueError("'duration' column not found in Train.csv. Available columns: " + str(df.columns.tolist()))
    
    # Check for NaN values in duration before sampling
    duration_nans = df['duration'].isna().sum()
    if duration_nans > 0:
        print(f"  ⚠ WARNING: Found {duration_nans} NaN values in duration - removing rows with NaN duration")
        df = df.dropna(subset=['duration'])
    
    # Ensure duration is numeric
    if not pd.api.types.is_numeric_dtype(df['duration']):
        print(f"  Converting duration from {df['duration'].dtype} to numeric...")
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        df = df.dropna(subset=['duration'])
    
    print(f"  ✓ Duration column validated: {len(df):,} valid rows")
    print(f"  ✓ Duration range: [{df['duration'].min():.1f}, {df['duration'].max():.1f}] seconds")
    
    # Sample - use 20% for faster tuning while maintaining quality
    taxi = df.sample(frac=1, random_state=42).copy()
    print(f"\nUsing 20% of training data: {len(taxi):,} rows (out of {len(df):,} total)")
    
    # Feature engineering
    print("\nApplying feature engineering...")
    taxi = engineer_features(taxi)
    
    # Define all features
    all_feature_columns = [
        'start_hour', 'start_minute', 'start_day', 'start_month',
        'dayOfWeek', 'timeOfDay', 'season',
        'is_weekend', 'is_rush_hour', 'is_night', 'is_holiday',
        'x2xDistance', 'y2yDistance',
        'manhattan_distance', 'euclidean_distance',
        'manhattan_distance_squared', 'euclidean_distance_squared',
        'euclidean_distance_cubed',
        'distance_category',
        'NumberOfPassengers', 'additionalStop',
        'solo_passenger', 'many_passengers',
        # 'distance_x_hour', 'distance_x_hour_squared',
        'distance_x_timeOfDay', 'distance_x_weekend',
        # 'distance_x_passengers',
        'passengers_x_weekend',
        'hour_x_is_weekend',
        'hour_sin', 'hour_cos',
        'month_sin', 'month_cos',
        'dayOfWeek_sin', 'dayOfWeek_cos',
        'minutes_since_midnight', 'minutes_until_midnight',
        'pickup_quadrant_x', 'pickup_quadrant_y',
        'dropoff_quadrant_x', 'dropoff_quadrant_y',
        'cross_quadrant_trip',
        'pickup_x', 'pickup_y', 'dropoff_x', 'dropoff_y'
    ]
    
    # Validate 'duration' is NOT in features - it's the target variable
    if 'duration' in all_feature_columns:
        raise ValueError("ERROR: 'duration' was included in all_feature_columns! It should only be the target.")
    
    X = taxi[all_feature_columns]
    y = taxi['duration']
    
    # Validate target and features
    print(f"\nPreparing X (features) and y (duration target):")
    print(f"  Features: {len(all_feature_columns)} features | Shape: {X.shape}")
    print(f"  Target (duration): {len(y)} samples | Shape: {y.shape}")
    
    # Ensure no NaN in features or target
    initial_samples = len(X)
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < initial_samples:
        removed = initial_samples - len(X)
        print(f"  ⚠ Removed {removed} rows with NaN values in features or duration")
    
    print(f"  ✓ Valid samples: {len(X):,}")
    print(f"\nDuration (target) statistics:")
    print(f"  Mean: {y.mean():.2f}s | Median: {y.median():.2f}s | Std: {y.std():.2f}s")
    print(f"  Min: {y.min():.2f}s | Max: {y.max():.2f}s")
    
    # ========== TUNE XGBOOST ==========
    print(f"\n\n{'#'*70}")
    print("XGBOOST HYPERPARAMETER TUNING")
    print(f"{'#'*70}")
    print(f"Target variable: duration (continuous regression)")
    print(f"Scoring metric: R² (coefficient of determination)\n")
    
    tuning_results = tune_xgboost(
        X, y, all_feature_columns,
        cv_splits=5,
        n_iter=50,
        random_state=42
    )
    
    best_model = tuning_results['model']
    importance_df, high_importance_features = analyze_feature_importance(
        best_model, all_feature_columns, importance_threshold=0.01
    )
    
    save_tuning_results('tuned_xgboost', tuning_results, importance_df)
    
    # ========== VALIDATION ==========
    print(f"\n\n{'='*70}")
    print("VALIDATING MODEL PREDICTIONS FOR DURATION (5-FOLD CV VERIFICATION)")
    print(f"{'='*70}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
    
    # Training predictions
    y_pred = best_model.predict(X)
    train_r2 = r2_score(y, y_pred)
    train_mse = mean_squared_error(y, y_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y, y_pred)
    train_mape = mean_absolute_percentage_error(y, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  CV R² (5-fold):")
    print(f"    Scores: {[f'{s:.6f}' for s in cv_scores]}")
    print(f"    Mean: {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})")
    print(f"  Training Performance:")
    print(f"    R² Score: {train_r2:.6f}")
    print(f"    MSE: {train_mse:.2f} seconds²")
    print(f"    RMSE: {train_rmse:.2f} seconds")
    print(f"    MAE: {train_mae:.2f} seconds")
    print(f"    MAPE: {train_mape:.4f} ({train_mape*100:.2f}%)")
    
    # Sample predictions
    sample_indices = np.random.choice(len(y), size=min(5, len(y)), replace=False)
    print(f"  Sample Predictions (actual duration vs predicted):")
    for idx in sorted(sample_indices):
        actual = y.iloc[idx]
        predicted = y_pred[idx]
        error_pct = abs(actual - predicted) / actual * 100
        print(f"    Actual: {actual:7.1f}s | Predicted: {predicted:7.1f}s | Error: {error_pct:5.1f}%")
    
    print(f"\n✓ Model confirmed to be predicting 'duration' (continuous regression)")
    
    # ========== FINAL RESULTS & RECOMMENDATION ==========
    print(f"\n\n{'='*70}")
    print("MODEL RESULTS & RECOMMENDATION")
    print(f"{'='*70}")
    print(f"\nTarget Variable: 'duration' (taxi trip duration in seconds)")
    print(f"Model Task: Continuous Regression")
    print(f"Evaluation Metrics: R², MSE, MAE\n")
    
    print("Model Performance Summary:")
    print(f"  CV R² (Mean): {cv_scores.mean():.6f} (+/- {cv_scores.std():.6f})")
    print(f"  Train R²:     {train_r2:.6f}")
    print(f"  Train MSE:    {train_mse:.2f} seconds²")
    print(f"  Train MAE:    {train_mae:.2f} seconds")
    print(f"  Train RMSE:   {train_rmse:.2f} seconds")
    print(f"  Train MAPE:   {train_mape:.4f} ({train_mape*100:.2f}%)")
    
    print(f"\nBest Hyperparameters:")
    for param, value in tuning_results['best_params'].items():
        print(f"  {param}: {value}")
    
    print(f"\n{'='*70}")
    print("Model training complete and ready for final evaluation!")
    print(f"{'='*70}")
