"""
Build XGBoost Model using Pass 1 Best Parameters and Generate Predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import json
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold, cross_validate

# ============== FEATURE ENGINEERING ==============

def parseDateTime(dateString):
    """Parse datetime string and return datetime object"""
    if pd.isna(dateString):
        raise ValueError(f"dateString is NaN")
    try:
        s = str(dateString).strip()
        if ',' in s:
            s = s.split(',', 1)[0].strip()
        
        # Try multiple datetime formats
        formats = [
            '%Y-%m-%d %H:%M:%S',  # Training data format
            '%m/%d/%y %H:%M',     # Test data format
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        
        # If no format matched, raise error with all attempts
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

# ============== MAIN EXECUTION ==============

if __name__ == "__main__":
    
    print("="*70)
    print("XGBOOST FINAL MODEL - PASS 1 PARAMETERS")
    print("="*70)
    
    # Load training data
    print("\nLoading Train.csv...")
    train_df = pd.read_csv('Train.csv')
    print(f"✓ Loaded {len(train_df):,} training rows")
    
    # Validate duration column
    print("\nValidating 'duration' target variable...")
    if 'duration' not in train_df.columns:
        raise ValueError("'duration' column not found in Train.csv")
    
    duration_nans = train_df['duration'].isna().sum()
    if duration_nans > 0:
        print(f"  ⚠ Removing {duration_nans} rows with NaN duration")
        train_df = train_df.dropna(subset=['duration'])
    
    if not pd.api.types.is_numeric_dtype(train_df['duration']):
        train_df['duration'] = pd.to_numeric(train_df['duration'], errors='coerce')
        train_df = train_df.dropna(subset=['duration'])
    
    # Set negative duration values to zero
    negative_count = (train_df['duration'] < 0).sum()
    if negative_count > 0:
        print(f"  ⚠ Found {negative_count} negative duration values, setting to zero")
        train_df.loc[train_df['duration'] < 0, 'duration'] = 0
    
    print(f"  ✓ Valid training rows: {len(train_df):,}")
    print(f"  ✓ Duration range: [{train_df['duration'].min():.1f}, {train_df['duration'].max():.1f}] seconds")
    
    # Feature engineering on training data
    print("\nEngineering features on training data...")
    taxi_train = engineer_features(train_df)
    
    # Define all features (same as in xgboost_tuning_2pass.py)
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
        'distance_x_hour', 'distance_x_hour_squared',
        'distance_x_timeOfDay', 'distance_x_weekend',
        'distance_x_passengers',
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
    
    X_train = taxi_train[all_feature_columns]
    y_train = taxi_train['duration']
    
    # Ensure no NaN values
    initial_samples = len(X_train)
    valid_mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    if len(X_train) < initial_samples:
        removed = initial_samples - len(X_train)
        print(f"  ⚠ Removed {removed} rows with NaN values")
    
    print(f"  ✓ Valid training samples: {len(X_train):,}")
    print(f"  ✓ Features used: {len(all_feature_columns)}")
    
    # Load best parameters from pass 1
    print("\nLoading best parameters from Pass 1...")
    with open('./results_logs/xgboost_best_params_pass1_20260603_202508.json', 'r') as f:
        best_params = json.load(f)
    
    print(f"✓ Best parameters loaded:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Build final model with best parameters
    print("\nBuilding XGBoost model with best parameters...")
    model = xgb.XGBRegressor(
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method='hist',
        **best_params
    )
    
    # Perform 5-fold cross-validation with multiple metrics
    print("\nPerforming 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate multiple metrics
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv, 
        scoring={'r2': 'r2', 'rmse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'},
        n_jobs=-1
    )
    
    # Extract and convert scores
    r2_scores = cv_results['test_r2']
    rmse_scores = np.sqrt(-cv_results['test_rmse'])  # Convert neg_mse to RMSE
    mae_scores = -cv_results['test_mae']  # Convert neg_mae to MAE
    
    print(f"\n{'='*90}")
    print(f"5-FOLD CROSS-VALIDATION METRICS")
    print(f"{'='*90}")
    print(f"{'Fold':<8} {'R² Score':<18} {'RMSE (s)':<18} {'MAE (s)':<18}")
    print(f"{'-'*90}")
    
    for fold in range(5):
        print(f"Fold {fold+1:<2} {r2_scores[fold]:<18.6f} {rmse_scores[fold]:<18.2f} {mae_scores[fold]:<18.2f}")
    
    print(f"{'-'*90}")
    print(f"{'Mean':<8} {r2_scores.mean():<18.6f} {rmse_scores.mean():<18.2f} {mae_scores.mean():<18.2f}")
    print(f"{'Std':<8} {r2_scores.std():<18.6f} {rmse_scores.std():<18.2f} {mae_scores.std():<18.2f}")
    print(f"{'='*90}\n")
    
    # Train final model on all training data for predictions
    print("Training final model on all training data for predictions...")
    model.fit(X_train, y_train)
    
    # Generate sample predictions on training data to show error metrics
    print("\nGenerating sample predictions on training data for error analysis...")
    sample_indices = np.random.choice(len(X_train), size=min(10, len(X_train)), replace=False)
    X_sample = X_train.iloc[sample_indices]
    y_sample_actual = y_train.iloc[sample_indices]
    y_sample_pred = model.predict(X_sample)
    
    # Calculate error percentages
    error_pct = np.abs(y_sample_actual.values - y_sample_pred) / y_sample_actual.values * 100
    
    print(f"\n{'='*90}")
    print(f"SAMPLE PREDICTIONS - 10 TEST DURATION VALUES WITH ERROR %")
    print(f"{'='*90}")
    print(f"{'Index':<8} {'Actual (s)':<15} {'Predicted (s)':<15} {'Error %':<15} {'Abs Error (s)':<15}")
    print(f"{'-'*90}")
    
    for i, (idx, actual, pred, err_pct) in enumerate(zip(sample_indices, y_sample_actual.values, y_sample_pred, error_pct)):
        abs_error = abs(actual - pred)
        print(f"{i+1:<8} {actual:<15.2f} {pred:<15.2f} {err_pct:<15.2f} {abs_error:<15.2f}")
    
    # Calculate aggregate error metrics for sample
    sample_rmse = np.sqrt(mean_squared_error(y_sample_actual, y_sample_pred))
    sample_mae = mean_absolute_error(y_sample_actual, y_sample_pred)
    sample_r2 = r2_score(y_sample_actual, y_sample_pred)
    avg_error_pct = error_pct.mean()
    
    print(f"{'-'*90}")
    print(f"{'Average Error %:':<38} {avg_error_pct:.2f}%")
    print(f"{'Sample RMSE (s):':<38} {sample_rmse:.2f}")
    print(f"{'Sample MAE (s):':<38} {sample_mae:.2f}")
    print(f"{'Sample R² Score:':<38} {sample_r2:.6f}")
    print(f"{'='*90}\n")
    
    # Ask for user input to continue
    print("Ready to generate predictions on Gotham Test Set?")
    user_input = input("Press Enter to continue (or type 'exit' to quit): ").strip().lower()
    
    if user_input == 'exit':
        print("Exiting without generating predictions.")
        exit()
    
    # Load test data
    print("\nLoading Gotham_Test_Set.csv...")
    gotham_df = pd.read_csv('./test_files/Gotham_Test_Set.csv')
    print(f"✓ Loaded {len(gotham_df):,} test rows")
    
    # Store original data for output
    original_gotham = gotham_df.copy()
    
    # Feature engineering on test data
    print("Engineering features on test data...")
    taxi_test = engineer_features(gotham_df)
    
    # Prepare test features - ensure same order and handling as training data
    X_test = taxi_test[all_feature_columns]
    
    # Handle NaN values in test data
    print(f"Checking for NaN values in test features...")
    nan_mask = X_test.isna().any(axis=1)
    nan_count = nan_mask.sum()
    
    if nan_count > 0:
        print(f"  ⚠ Found {nan_count} rows with NaN values in features")
        print(f"  Rows with NaN will be kept but predictions may be affected")
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    print(f"✓ Generated {len(y_pred)} predictions")
    
    # Set negative predictions to zero
    negative_pred_count = (y_pred < 0).sum()
    if negative_pred_count > 0:
        print(f"  ⚠ Found {negative_pred_count} negative predictions, setting to zero")
        y_pred = np.maximum(y_pred, 0)
    
    # Create output dataframe with predictions
    print("\nPreparing output file...")
    output_df = original_gotham.copy()
    
    # Ensure the order and structure match the input
    output_df['duration'] = y_pred
    
    # Round predictions to match potential rounding in training data
    output_df['duration'] = output_df['duration'].round(2)
    
    # Save to CSV
    output_filename = './test_files/Gotham_Test_Set_with_predictions.csv'
    output_df.to_csv(output_filename, index=False)
    print(f"✓ Saved predictions to: {output_filename}")
    
    print(f"\n{'='*70}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*70}")
    print(f"Test samples: {len(output_df):,}")
    print(f"Predictions - Mean: {y_pred.mean():.2f}s | Median: {np.median(y_pred):.2f}s")
    print(f"Predictions - Min: {y_pred.min():.2f}s | Max: {y_pred.max():.2f}s")
    print(f"Predictions - Std: {y_pred.std():.2f}s")
    print(f"{'='*70}\n")
    
    print("✓ Process completed successfully!")
