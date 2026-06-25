"""
Forest Analysis - Multi-Model Comparison with Training/Test Logs
Compares: Random Forest, Extra Trees, Decision Tree with Bagging, XGBoost, LightGBM
Uses 50% of training data with separate train/test splits for detailed analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import json
import os
import time
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, 
    BaggingRegressor, AdaBoostRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# XGBoost with GPU support
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available")

# LightGBM with GPU support
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("Warning: LightGBM not available")

# ============== FEATURE ENGINEERING FUNCTIONS ==============

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
        return 4  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    elif month in [9, 10, 11]:
        return 3  # Fall
    else:
        raise ValueError(f"Invalid month value: {month}")

def isWeekend(day_of_week):
    if day_of_week is None:
        raise ValueError(f"day_of_week is None")
    return int(day_of_week >= 5)

def getTimeofDay(start_hour):
    if start_hour < 8:
        return 0  # Early morning
    elif start_hour < 11:
        return 1  # Morning
    elif start_hour < 13:
        return 2  # Lunch
    elif start_hour < 16:
        return 3  # Afternoon
    elif start_hour < 18:
        return 4  # Evening
    else:
        return 5  # Night

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
    """Enhanced feature engineering for taxi duration prediction"""
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
    
    # Passenger categories
    taxi['solo_passenger'] = (taxi['NumberOfPassengers'] == 1).astype(int)
    taxi['many_passengers'] = (taxi['NumberOfPassengers'] > 3).astype(int)
    
    # Time-based aggregations
    taxi['minutes_since_midnight'] = taxi['start_hour'] * 60 + taxi['start_minute']
    taxi['minutes_until_midnight'] = 24 * 60 - taxi['minutes_since_midnight']
    
    # Interaction: Passengers × Is_Weekend
    taxi['passengers_x_weekend'] = taxi['NumberOfPassengers'] * taxi['is_weekend']
    
    # Interaction: Time of Day × Season
    taxi['timeOfDay_x_season'] = taxi['timeOfDay'] * taxi['season']
    
    # Start point quadrant
    taxi['pickup_quadrant_x'] = (taxi['pickup_x'] > taxi['pickup_x'].median()).astype(int)
    taxi['pickup_quadrant_y'] = (taxi['pickup_y'] > taxi['pickup_y'].median()).astype(int)
    
    # Destination quadrant
    taxi['dropoff_quadrant_x'] = (taxi['dropoff_x'] > taxi['dropoff_x'].median()).astype(int)
    taxi['dropoff_quadrant_y'] = (taxi['dropoff_y'] > taxi['dropoff_y'].median()).astype(int)
    
    # Cross-quadrant trips
    taxi['cross_quadrant_trip'] = (
        (taxi['pickup_quadrant_x'] != taxi['dropoff_quadrant_x']) |
        (taxi['pickup_quadrant_y'] != taxi['dropoff_quadrant_y'])
    ).astype(int)
    
    return taxi

# ============== MODEL EVALUATION WITH TRAIN/TEST SPLIT ==============

def evaluate_model_cv(model, X, y, k=5, model_name="Model"):
    """
    Evaluate model using k-fold cross-validation
    
    Args:
        model: Model to evaluate
        X: Feature matrix
        y: Target values
        k: Number of folds
        model_name: Name of the model
    
    Returns:
        Dictionary with CV metrics
    """
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Get R² scores
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    
    return {
        'name': model_name,
        'model': model,
        'cv_r2_scores': r2_scores,
        'mean_r2': r2_scores.mean(),
        'std_r2': r2_scores.std(),
    }

def compare_models(models_dict, X, y, k=5):
    """
    Compare multiple models using k-fold cross-validation
    
    Args:
        models_dict: Dictionary of {model_name: model_instance}
        X: Feature matrix
        y: Target values
        k: Number of folds for cross-validation
    
    Returns:
        List of result dictionaries sorted by mean R²
    """
    
    results = []
    
    print(f"\nRunning {k}-fold cross-validation...")
    print("-" * 80)
    
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name}...", end=" ", flush=True)
        
        # Create a fresh model instance
        model_copy = type(model)(**model.get_params())
        
        result = evaluate_model_cv(
            model_copy, X, y, k=k, model_name=model_name
        )
        
        results.append(result)
        print(f"Mean R² = {result['mean_r2']:.6f} (±{result['std_r2']:.6f})")
    
    # Sort by mean R²
    results_sorted = sorted(results, key=lambda x: x['mean_r2'], reverse=True)
    
    return results_sorted

# ============== LOGGING FUNCTIONS ==============

def ensure_log_dir(log_dir='./results_logs'):
    """Create logs directory if it doesn't exist"""
    Path(log_dir).mkdir(exist_ok=True)
    return log_dir

def save_comparison_results(results, log_dir='./results_logs'):
    """
    Save model comparison results to JSON and CSV files
    
    Args:
        results: List of result dictionaries from compare_models
        log_dir: Directory to save logs
    
    Returns:
        Tuple of (csv_path, json_path)
    """
    log_dir = ensure_log_dir(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create comparison DataFrame
    comparison_data = []
    for r in results:
        comparison_data.append({
            'Model': r['name'],
            'Mean CV R²': f"{r['mean_r2']:.6f}",
            'Std CV R²': f"{r['std_r2']:.6f}",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save as CSV
    csv_path = os.path.join(log_dir, f'forest_analysis_cv_comparison_{timestamp}.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison results to: {csv_path}")
    
    # Save as JSON with detailed metrics
    json_data = {
        'timestamp': timestamp,
        'evaluation_method': 'k-fold cross-validation',
        'models': []
    }
    
    for r in results:
        model_data = {
            'name': r['name'],
            'mean_r2': float(r['mean_r2']),
            'std_r2': float(r['std_r2']),
            'cv_scores': [float(s) for s in r['cv_r2_scores']],
        }
        json_data['models'].append(model_data)
    
    json_path = os.path.join(log_dir, f'forest_analysis_cv_detailed_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved detailed results to: {json_path}")
    
    return csv_path, json_path, comparison_df

# ============== VISUALIZATION ==============

def plot_model_comparison(results, save_dir='./results_logs'):
    """Plot model comparison with CV R² scores and error bars"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = [r['name'] for r in results]
    mean_r2 = [r['mean_r2'] for r in results]
    std_r2 = [r['std_r2'] for r in results]
    
    x = np.arange(len(model_names))
    
    bars = ax.bar(x, mean_r2, width=0.6, label='Mean CV R²', alpha=0.8, capsize=5, 
                  yerr=std_r2, error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, mean_r2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean_val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('Mean CV R² Score', fontweight='bold', fontsize=12)
    ax.set_title('Model Comparison - 5-Fold Cross-Validation', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'forest_analysis_cv_comparison_plot_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {save_path}")
    plt.close()

# ============== COMPLEXITY ANALYSIS (BIAS-VARIANCE TRADEOFF) ==============

def create_model_with_depth(model_class, depth, random_state=42):
    """
    Create a model instance with specified depth/complexity.
    Handles different model types that have different parameter names.
    
    Args:
        model_class: The model class to instantiate
        depth: Complexity parameter (max_depth for trees, n_layers for NNs, etc.)
        random_state: Random seed
        
    Returns:
        Instantiated model with depth parameter set
    """
    class_name = model_class.__name__
    
    # For models that take max_depth directly
    if class_name in ['DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor']:
        return model_class(max_depth=depth, random_state=random_state, n_jobs=-1)
    
    # For BaggingRegressor: wrap a DT with max_depth
    elif class_name == 'BaggingRegressor':
        base_estimator = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
        return model_class(estimator=base_estimator, n_estimators=100, random_state=random_state, n_jobs=-1)
    
    # For gradient boosting models
    elif class_name in ['GradientBoostingRegressor']:
        return model_class(n_estimators=100, learning_rate=0.1, max_depth=depth, random_state=random_state)
    
    # For XGBoost
    elif class_name == 'XGBRegressor':
        return model_class(n_estimators=100, learning_rate=0.1, max_depth=depth, random_state=random_state, n_jobs=-1, verbosity=0)
    
    # For LightGBM
    elif class_name == 'LGBMRegressor':
        return model_class(n_estimators=100, learning_rate=0.1, max_depth=depth, random_state=random_state, n_jobs=-1, verbose=-1)
    
    else:
        raise ValueError(f"Unknown model class: {class_name}")

def compute_and_save_complexity_analysis_multi(models_list, X, y, depths=range(1, 25), k=5, log_dir='./results_logs'):
    """
    Compute bias-variance tradeoff for multiple models by varying complexity (max_depth)
    Saves results to CSV with model names for comparison.
    **INCLUDES CHECKPOINTING**: Saves after each model completes to enable resuming from crashes.
    
    Args:
        models_list: List of tuples (model_name, model_class)
        X: Feature matrix
        y: Target values
        depths: Range of complexity values to test
        k: Number of CV folds
        log_dir: Directory to save logs
    
    Returns:
        Path to saved CSV file
    """
    log_dir = ensure_log_dir(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\n" + "=" * 80)
    print("COMPLEXITY ANALYSIS - BIAS-VARIANCE TRADEOFF")
    print("=" * 80)
    print("Computing train_error and val_error by varying model complexity (max_depth)...")
    print("-" * 80)
    
    # Checkpoint file to save results incrementally
    checkpoint_path = os.path.join(log_dir, f'complexity_analysis_checkpoint_{timestamp}.csv')
    all_results = []
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    
    for model_name, model_class in models_list:
        print(f"\nAnalyzing {model_name}...")
        model_results = []
        
        for depth in depths:
            try:
                # Create model with current depth (handles different model types)
                model = create_model_with_depth(model_class, depth, random_state=42)
                
                # Get CV scores for validation error
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
                val_error = 1 - cv_scores.mean()  # Convert R² to error
                
                # Fit on full data for training error
                model.fit(X, y)
                train_r2 = model.score(X, y)
                train_error = 1 - train_r2
                
                result = {
                    'model': model_name,
                    'complexity': depth,
                    'train_error': train_error,
                    'val_error': val_error
                }
                all_results.append(result)
                model_results.append(result)
                
                print(f"  Depth {depth:2d}: train_error = {train_error:.6f}, val_error = {val_error:.6f}")
                
            except Exception as e:
                print(f"  Depth {depth:2d}: ERROR - {str(e)}")
                continue
        
        # **CHECKPOINT**: Save after each model completes
        if model_results:
            df_checkpoint = pd.DataFrame(all_results)
            df_checkpoint.to_csv(checkpoint_path, index=False)
            print(f"  ✓ Checkpoint saved: {len(all_results)} results so far")
    
    # Final save with proper naming
    final_csv_path = os.path.join(log_dir, f'complexity_analysis_forest_{timestamp}.csv')
    df = pd.DataFrame(all_results)
    df.to_csv(final_csv_path, index=False)
    print(f"\n✓ Saved complexity analysis to: {final_csv_path}")
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    # Display results summary
    print("\n" + "-" * 80)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("-" * 80)
    for model_name, _ in models_list:
        model_data = df[df['model'] == model_name]
        if len(model_data) > 0:
            print(f"\n{model_name}:")
            print(f"  Complexity range: {model_data['complexity'].min()} - {model_data['complexity'].max()}")
            print(f"  Best test performance (lowest val_error):")
            best_idx = model_data['val_error'].idxmin()
            best_row = model_data.loc[best_idx]
            print(f"    Depth: {int(best_row['complexity'])}, train_error: {best_row['train_error']:.6f}, val_error: {best_row['val_error']:.6f}")
            print(f"  Overfitting gap at best depth: {best_row['train_error'] - best_row['val_error']:.6f}")
    
    return final_csv_path

def plot_complexity_analysis(csv_path, save_dir='./results_logs'):
    """
    Plot complexity analysis with train_error and val_error for each model
    
    Args:
        csv_path: Path to complexity analysis CSV file
        save_dir: Directory to save plot
    """
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, len(df['model'].unique()), figsize=(6 * len(df['model'].unique()), 5))
    
    # Handle case where there's only one model
    if len(df['model'].unique()) == 1:
        axes = [axes]
    
    for idx, (ax, model_name) in enumerate(zip(axes, df['model'].unique())):
        model_data = df[df['model'] == model_name].sort_values('complexity')
        
        ax.plot(model_data['complexity'], model_data['train_error'], 'b-o', linewidth=2.5, 
               label='Training Error', markersize=6, alpha=0.8)
        ax.plot(model_data['complexity'], model_data['val_error'], 'r-s', linewidth=2.5, 
               label='Validation Error', markersize=6, alpha=0.8)
        
        # Shade the gap between curves
        ax.fill_between(model_data['complexity'], model_data['train_error'], 
                       model_data['val_error'], alpha=0.15, color='gray', label='Generalization Gap')
        
        ax.set_xlabel('Model Complexity (max_depth)', fontweight='bold')
        ax.set_ylabel('Error (1 - R²)', fontweight='bold')
        ax.set_title(f'{model_name}\nBias-Variance Tradeoff', fontweight='bold', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'forest_analysis_complexity_{timestamp}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved complexity analysis plot to: {save_path}")
    plt.close()

# ============== MAIN EXECUTION ==============

if __name__ == "__main__":
    
    print("=" * 80)
    print("FOREST ANALYSIS - MULTI-MODEL COMPARISON")
    print("=" * 80)
    print("\nModels to compare:")
    print("  1. Random Forest")
    print("  2. Extra Trees")
    print("  3. Decision Tree with Bagging")
    print("  4. XGBoost (if available)")
    print("  5. LightGBM (if available)")
    
    # Load data
    print("\n" + "-" * 80)
    print("Loading data...")
    df = pd.read_csv('Train.csv')
    
    # Use 50% of training data
    taxi = df.sample(frac=0.5, random_state=42).copy()
    print(f"Using 50% of training data: {len(taxi):,} rows (out of {len(df):,} total)")
    
    # Feature engineering
    print("\nApplying feature engineering...")
    taxi = engineer_features(taxi)
    
    # Define feature set
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
        'distance_x_hour_squared',
        'distance_x_timeOfDay', 'distance_x_weekend',
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
    
    # Prepare features and target
    X = taxi[all_feature_columns]
    y = taxi['duration']
    
    # Handle NaN values
    initial_samples = len(X)
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    
    if len(X) < initial_samples:
        removed = initial_samples - len(X)
        print(f"⚠ Removed {removed} rows with NaN values")
    
    print(f"\nData prepared:")
    print(f"  Features: {len(all_feature_columns)} features")
    print(f"  Samples: {len(X):,} rows")
    print(f"  Target (duration): Min={y.min():.1f}s, Max={y.max():.1f}s, Mean={y.mean():.1f}s")
    
    # Initialize models
    print("\n" + "-" * 80)
    print("Initializing models...")
    print("-" * 80)
    
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1, verbose=0
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1, verbose=0
        ),
        'Decision Tree + Bagging': BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=20, random_state=42),
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }
    
    # Add XGBoost if available
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        )
        print("✓ XGBoost available and added to models")
    else:
        print("✗ XGBoost not available")
    
    # Add LightGBM if available
    if LGB_AVAILABLE:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
        print("✓ LightGBM available and added to models")
    else:
        print("✗ LightGBM not available")
    
    print(f"\nTotal models to evaluate: {len(models)}")
    
    # Run k-fold cross-validation comparison
    print("\n" + "=" * 80)
    print("EVALUATING MODELS WITH K-FOLD CROSS-VALIDATION")
    print("=" * 80)
    
    start_time = time.time()
    
    results = compare_models(
        models, X, y, 
        k=5
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Model evaluation completed in {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    
    # Display results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS (Sorted by Mean CV R²)")
    print("=" * 80)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['name']}")
        print(f"   Mean CV R²:  {r['mean_r2']:.6f} (±{r['std_r2']:.6f})")
        print(f"   CV Scores: {[f'{s:.6f}' for s in r['cv_r2_scores']]}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    csv_path, json_path, comparison_df = save_comparison_results(results, log_dir='./results_logs')
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_model_comparison(results, save_dir='./results_logs')
    
    # Complexity analysis for models with mean CV R² >= 0.76
    print("\n" + "=" * 80)
    print("FILTERING MODELS FOR COMPLEXITY ANALYSIS")
    print("=" * 80)
    print("Threshold: Mean CV R² >= 0.76")
    
    qualifying_models = []
    for r in results:
        if r['mean_r2'] >= 0.76:
            model_name = r['name']
            qualifying_models.append((model_name, r['mean_r2']))
            print(f"  ✓ {model_name}: {r['mean_r2']:.6f}")
        else:
            print(f"  ✗ {model_name}: {r['mean_r2']:.6f} (below threshold)")
    
    if len(qualifying_models) > 0:
        print(f"\n{len(qualifying_models)} model(s) qualify for complexity analysis")
        
        # Get model classes for qualifying models
        models_for_complexity = []
        for model_name, _ in qualifying_models:
            # Find the model class from the original models dict
            for orig_model_name, orig_model_instance in models.items():
                if orig_model_name == model_name:
                    models_for_complexity.append((model_name, type(orig_model_instance)))
                    break
        
        print("\n" + "=" * 80)
        print("RUNNING COMPLEXITY ANALYSIS")
        print("=" * 80)
        complexity_csv = compute_and_save_complexity_analysis_multi(
            models_for_complexity, X, y, 
            depths=range(1, 16), 
            k=5, 
            log_dir='./results_logs'
        )
        plot_complexity_analysis(complexity_csv, save_dir='./results_logs')
    else:
        print(f"\n✗ No models meet the R² >= 0.76 threshold for complexity analysis")
        print(f"  Best model: {results[0]['name']} with R² = {results[0]['avg_test_r2']:.6f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    best_model = results[0]
    print(f"\nBest Model (by CV R²): {best_model['name']}")
    print(f"  Mean CV R²:  {best_model['mean_r2']:.6f}")
    print(f"  Std CV R²:   {best_model['std_r2']:.6f}")
    
    print(f"\nResults saved to: ./results_logs/")
    print(f"  CSV: {os.path.basename(csv_path)}")
    print(f"  JSON: {os.path.basename(json_path)}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
