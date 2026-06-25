"""
XGBoost Two-Pass Tuning with Feature Selection
Pass 1: Full tuning on all features
Pass 2: Retune on only high-importance features
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

def tune_xgboost(X, y, feature_names, pass_num=1, cv_splits=5, n_iter=80, random_state=42):
    """
    Perform RandomizedSearchCV for XGBoost
    """
    
    # More aggressive search space for pass 1
    param_dist = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [4, 5, 6, 7, 8, 9, 10, 11],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.5, 1, 2],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0.5, 1, 2, 5],
    }
    
    print(f"\n{'='*70}")
    print(f"PASS {pass_num}: RandomizedSearchCV")
    print(f"{'='*70}")
    print(f"Features: {len(feature_names)} | Samples: {len(X):,}")
    print(f"Testing {n_iter} parameter combinations with {cv_splits}-fold CV")
    print(f"Total fits: {n_iter * cv_splits}")
    
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
        verbose=2
    )
    
    print(f"\nStarting fit at {datetime.now().strftime('%H:%M:%S')}...")
    random_search.fit(X, y)
    
    elapsed = time.time() - start_time
    print(f"\nFit completed in {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    
    # Results
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_
    
    print(f"\n{'='*70}")
    print(f"PASS {pass_num} RESULTS")
    print(f"{'='*70}")
    print(f"Best CV R² Score: {best_cv_score:.6f}")
    print(f"\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Top 10 results
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    print(f"\nTop 10 Configurations:")
    print(results_df[['rank_test_score', 'mean_test_score', 'std_test_score']].head(10).to_string())
    
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
    
    print(f"\n\nFeatures with importance >= {importance_threshold}:")
    print(f"  Count: {len(high_importance)}")
    print(f"  Total importance: {high_importance['Importance'].sum():.4f}")
    
    print(f"\nFeatures with importance < {importance_threshold}:")
    print(f"  Count: {len(low_importance)}")
    print(f"  Total importance: {low_importance['Importance'].sum():.4f}")
    print(f"\nCandidates for removal:")
    print(low_importance.to_string(index=False))
    
    return importance_df, high_importance['Feature'].tolist()

def save_tuning_results(pass_num, tuning_results, importance_df, log_dir='./results_logs'):
    """Save tuning results"""
    Path(log_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save best parameters
    params_path = os.path.join(log_dir, f'xgboost_best_params_pass{pass_num}_{timestamp}.json')
    with open(params_path, 'w') as f:
        json.dump(tuning_results['best_params'], f, indent=2)
    print(f"Saved best parameters to: {params_path}")
    
    # Save feature importance
    if importance_df is not None:
        importance_path = os.path.join(log_dir, f'feature_importance_pass{pass_num}_{timestamp}.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"Saved feature importance to: {importance_path}")
    
    return params_path

# ============== MAIN EXECUTION ==============

if __name__ == "__main__":
    
    print("="*70)
    print("XGBOOST TWO-PASS TUNING WITH FEATURE SELECTION")
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
    
    # Sample - use full 30% for more thorough testing
    taxi = df.sample(frac=1, random_state=42).copy()
    print(f"\nUsing 30% of training data: {len(taxi):,} rows (out of {len(df):,} total)")
    
    # Feature engineering
    print("\nApplying feature engineering...")
    taxi = engineer_features(taxi)
    
    # Define all features
    # NOTE: 'duration' is deliberately excluded - it's the target variable, not a feature
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
    
    # ========== PASS 1: Full tuning on all features ==========
    print(f"\n\n{'#'*70}")
    print("PASS 1: COMPREHENSIVE TUNING ON ALL FEATURES")
    print(f"{'#'*70}")
    print(f"Target variable: duration (continuous regression)")
    print(f"Scoring metric: R² (coefficient of determination)\n")
    
    tuning_results_pass1 = tune_xgboost(
        X, y, all_feature_columns,
        pass_num=1,
        cv_splits=5,
        n_iter=80,
        random_state=42
    )
    
    best_model_pass1 = tuning_results_pass1['model']
    importance_df_pass1, high_importance_features = analyze_feature_importance(
        best_model_pass1, all_feature_columns, importance_threshold=0.01
    )
    
    save_tuning_results(1, tuning_results_pass1, importance_df_pass1)
    
    print(f"\nPass 1 Summary:")
    print(f"  Best CV R²: {tuning_results_pass1['best_cv_score']:.6f}")
    print(f"  Features with importance >= 0.01: {len(high_importance_features)}")
    print(f"  Runtime: {tuning_results_pass1['elapsed_minutes']:.1f} minutes")
    
    # ========== PASS 2: Retune with only high-importance features ==========
    print(f"\n\n{'#'*70}")
    print("PASS 2: RETUNE WITH HIGH-IMPORTANCE FEATURES ONLY")
    print(f"{'#'*70}")
    print(f"Target variable: duration (continuous regression)")
    print(f"Scoring metric: R² (coefficient of determination)\n")
    
    # Subset data to high-importance features
    X_filtered = X[high_importance_features]
    
    removed_features = set(all_feature_columns) - set(high_importance_features)
    print(f"Features removed: {len(removed_features)}")
    print(f"Features retained: {len(high_importance_features)}")
    print(f"Removed features: {removed_features}\n")
    
    tuning_results_pass2 = tune_xgboost(
        X_filtered, y, high_importance_features,
        pass_num=2,
        cv_splits=5,
        n_iter=80,
        random_state=42
    )
    
    best_model_pass2 = tuning_results_pass2['model']
    importance_df_pass2, _ = analyze_feature_importance(
        best_model_pass2, high_importance_features, importance_threshold=0.01
    )
    
    save_tuning_results(2, tuning_results_pass2, importance_df_pass2)
    
    print(f"\nPass 2 Summary:")
    print(f"  Best CV R²: {tuning_results_pass2['best_cv_score']:.6f}")
    print(f"  Runtime: {tuning_results_pass2['elapsed_minutes']:.1f} minutes")
    
    # ========== VALIDATION ON BOTH MODELS ==========
    print(f"\n\n{'='*70}")
    print("VALIDATING MODEL PREDICTIONS FOR DURATION")
    print(f"{'='*70}")
    
    # Pass 1 predictions
    y_pred_pass1 = best_model_pass1.predict(X)
    train_r2_pass1 = r2_score(y, y_pred_pass1)
    train_rmse_pass1 = np.sqrt(mean_squared_error(y, y_pred_pass1))
    train_mae_pass1 = mean_absolute_error(y, y_pred_pass1)
    train_mape_pass1 = mean_absolute_percentage_error(y, y_pred_pass1)
    
    print(f"\nPass 1 - Model Performance on Training Data (predicting duration):")
    print(f"  R² Score: {train_r2_pass1:.6f}")
    print(f"  RMSE: {train_rmse_pass1:.2f} seconds")
    print(f"  MAE: {train_mae_pass1:.2f} seconds")
    print(f"  MAPE: {train_mape_pass1:.4f} ({train_mape_pass1*100:.2f}%)")
    
    # Pass 2 predictions
    y_pred_pass2 = best_model_pass2.predict(X_filtered)
    train_r2_pass2 = r2_score(y, y_pred_pass2)
    train_rmse_pass2 = np.sqrt(mean_squared_error(y, y_pred_pass2))
    train_mae_pass2 = mean_absolute_error(y, y_pred_pass2)
    train_mape_pass2 = mean_absolute_percentage_error(y, y_pred_pass2)
    
    print(f"\nPass 2 - Model Performance on Training Data (predicting duration):")
    print(f"  R² Score: {train_r2_pass2:.6f}")
    print(f"  RMSE: {train_rmse_pass2:.2f} seconds")
    print(f"  MAE: {train_mae_pass2:.2f} seconds")
    print(f"  MAPE: {train_mape_pass2:.4f} ({train_mape_pass2*100:.2f}%)")
    
    # Sample predictions from best model
    best_model = best_model_pass2 if tuning_results_pass2['best_cv_score'] > tuning_results_pass1['best_cv_score'] else best_model_pass1
    best_predictions = y_pred_pass2 if tuning_results_pass2['best_cv_score'] > tuning_results_pass1['best_cv_score'] else y_pred_pass1
    
    sample_indices = np.random.choice(len(y), size=min(5, len(y)), replace=False)
    print(f"\nSample Predictions from Best Model (actual duration vs predicted):")
    for idx in sorted(sample_indices):
        actual = y.iloc[idx]
        predicted = best_predictions[idx]
        error_pct = abs(actual - predicted) / actual * 100
        print(f"  Actual: {actual:7.1f}s | Predicted: {predicted:7.1f}s | Error: {error_pct:5.1f}%")
    
    print(f"\n✓ Both models confirmed to be predicting 'duration' (continuous regression)")
    
    # ========== FINAL COMPARISON ==========
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON & FEATURE ENGINEERING IMPACT")
    print(f"{'='*70}")
    print(f"\nTarget Variable: 'duration' (taxi trip duration in seconds)")
    print(f"Model Task: Continuous Regression\n")
    
    print(f"Pass 1 (All {len(all_feature_columns)} features):")
    print(f"  CV R²: {tuning_results_pass1['best_cv_score']:.6f}")
    print(f"  Train R² (verification): {train_r2_pass1:.6f}")
    print(f"  Best Depth: {tuning_results_pass1['best_params'].get('max_depth')}")
    print(f"  Best Learning Rate: {tuning_results_pass1['best_params'].get('learning_rate')}")
    print(f"  Runtime: {tuning_results_pass1['elapsed_minutes']:.1f} min")
    
    print(f"\nPass 2 ({len(high_importance_features)} high-importance features):")
    print(f"  CV R²: {tuning_results_pass2['best_cv_score']:.6f}")
    print(f"  Train R² (verification): {train_r2_pass2:.6f}")
    print(f"  Best Depth: {tuning_results_pass2['best_params'].get('max_depth')}")
    print(f"  Best Learning Rate: {tuning_results_pass2['best_params'].get('learning_rate')}")
    print(f"  Runtime: {tuning_results_pass2['elapsed_minutes']:.1f} min")
    
    improvement_cv = tuning_results_pass2['best_cv_score'] - tuning_results_pass1['best_cv_score']
    improvement_train = train_r2_pass2 - train_r2_pass1
    
    print(f"\nFeature Engineering Impact:")
    print(f"  CV R² improvement: {improvement_cv:+.6f}")
    print(f"  Train R² improvement: {improvement_train:+.6f}")
    print(f"  Features removed: {len(removed_features)} ({len(removed_features)/len(all_feature_columns)*100:.1f}%)")
    
    if improvement_cv > 0.001:  # More than 0.1% improvement
        print(f"\n✓ Feature selection IMPROVED the model by {abs(improvement_cv):.6f}!")
        print(f"  Use Pass 2 hyperparameters for final training")
        print(f"  Optimal features ({len(high_importance_features)}): {high_importance_features}")
    elif improvement_cv > -0.001:  # Within 0.1%
        print(f"\n≈ Feature selection had minimal impact (change: {improvement_cv:+.6f})")
        print(f"  Either approach is acceptable")
        print(f"  Pass 2 is preferred (fewer features = faster inference)")
    else:
        print(f"\n✗ Feature selection decreased performance by {abs(improvement_cv):.6f}")
        print(f"  Use Pass 1 hyperparameters for final training")
        print(f"  Use all {len(all_feature_columns)} features")
    
    print(f"\n{'='*70}")
