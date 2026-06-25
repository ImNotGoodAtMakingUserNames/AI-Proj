
"""
Taxi Duration Prediction - Model Optimization Script
Goal: Achieve R² >= 0.9 using tree-based methods with ensemble techniques
Approach: 30% data sample, 5-fold CV, feature importance analysis, GPU acceleration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import json
import os
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Visualization imports
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

# CatBoost with GPU support
try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False
    print("Warning: CatBoost not available")

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
    """Extract hour from pickup_datetime"""
    return int(parseDateTime(dateString).hour)

def getStartMinute(dateString):
    """Extract minute from pickup_datetime"""
    return int(parseDateTime(dateString).minute)

def getStartSecond(dateString):
    """Extract second from pickup_datetime"""
    return int(parseDateTime(dateString).second)

def getStartDay(dateString):
    """Extract day from pickup_datetime"""
    return int(parseDateTime(dateString).day)

def getStartMonth(dateString):
    """Extract month from pickup_datetime"""
    return int(parseDateTime(dateString).month)

def getStartYear(dateString):
    """Extract year from pickup_datetime"""
    return int(parseDateTime(dateString).year)

def getDayOfWeek(start_month, start_day, start_year):
    """Get day of week from individual datetime components"""
    if any(pd.isna(x) for x in [start_month, start_day, start_year]):
        raise ValueError(f"One or more datetime components are NaN")
    try:
        dt = datetime(int(start_year), int(start_month), int(start_day))
        return int(dt.weekday())
    except Exception as e:
        raise ValueError(f"Failed to create datetime from components: {e}")

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
        raise ValueError(f"Invalid month value: {month}")

def isWeekend(day_of_week):
    """Check if the day is a weekend (Saturday=5, Sunday=6)"""
    if day_of_week is None:
        raise ValueError(f"day_of_week is None")
    return int(day_of_week >= 5)

def getTimeofDay(start_hour):
    """Categorize time of day"""
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
    """Check if the date is a US federal holiday (configured for 2034)"""
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

# ============== FEATURE ENGINEERING ==============

def engineer_features(df):
    """
    Apply all feature engineering transformations
    Returns DataFrame with engineered features
    """
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
    
    # Polynomial features for distance
    taxi['x2xDistance_squared'] = taxi['x2xDistance'] ** 2
    taxi['y2yDistance_squared'] = taxi['y2yDistance'] ** 2
    taxi['x2xDistance_cubed'] = taxi['x2xDistance'] ** 3
    taxi['y2yDistance_cubed'] = taxi['y2yDistance'] ** 3
    
    # Interaction terms
    taxi['distance_x_timeOfDay'] = taxi['x2xDistance'] * taxi['timeOfDay']
    taxi['distance_y_timeOfDay'] = taxi['y2yDistance'] * taxi['timeOfDay']
    taxi['distance_x_weekend'] = taxi['x2xDistance'] * taxi['is_weekend']
    taxi['distance_y_weekend'] = taxi['y2yDistance'] * taxi['is_weekend']
    taxi['distance_x_hour'] = taxi['x2xDistance'] * taxi['start_hour']
    taxi['distance_y_hour'] = taxi['y2yDistance'] * taxi['start_hour']
    
    # Cyclic encodings for periodic features
    taxi['hour_sin'] = np.sin(2 * np.pi * taxi['start_hour'] / 24)
    taxi['hour_cos'] = np.cos(2 * np.pi * taxi['start_hour'] / 24)
    taxi['month_sin'] = np.sin(2 * np.pi * taxi['start_month'] / 12)
    taxi['month_cos'] = np.cos(2 * np.pi * taxi['start_month'] / 12)
    taxi['dayOfWeek_sin'] = np.sin(2 * np.pi * taxi['dayOfWeek'] / 7)
    taxi['dayOfWeek_cos'] = np.cos(2 * np.pi * taxi['dayOfWeek'] / 7)
    
    # Distance metrics
    taxi['manhattan_distance'] = taxi['x2xDistance'] + taxi['y2yDistance']
    taxi['manhattan_distance_squared'] = taxi['manhattan_distance'] ** 2
    taxi['total_distance'] = np.sqrt(taxi['x2xDistance'] ** 2 + taxi['y2yDistance'] ** 2)
    
    # Distance categories
    taxi['distance_category'] = pd.cut(
        taxi['total_distance'], 
        bins=[0, 1, 2, 5, 10, float('inf')], 
        labels=['very_short', 'short', 'medium', 'long', 'very_long']
    ).cat.codes
    
    # Destination interactions
    taxi['distance_x_dest_x'] = taxi['x2xDistance'] * taxi['dropoff_x']
    taxi['distance_y_dest_y'] = taxi['y2yDistance'] * taxi['dropoff_y']
    
    # Time interactions
    taxi['hour_x_is_weekend'] = taxi['start_hour'] * taxi['is_weekend']
    taxi['timeOfDay_x_season'] = taxi['timeOfDay'] * taxi['season']
    
    return taxi

# ============== MODEL EVALUATION ==============

def evaluate_model_cv(model, X, y, k=5, model_name="Model"):
    """
    Evaluate model using k-fold cross-validation
    Returns dictionary with CV scores and mean R²
    """
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    
    return {
        'name': model_name,
        'cv_scores': cv_scores,
        'mean_r2': cv_scores.mean(),
        'std_r2': cv_scores.std(),
        'model': model
    }

def compare_models(models_dict, X, y, k=5):
    """
    Compare multiple models using k-fold CV
    Returns sorted DataFrame by mean R² and sorted results list
    """
    results = []
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...", end=" ", flush=True)
        result = evaluate_model_cv(model, X, y, k=k, model_name=model_name)
        results.append(result)
        print(f"Mean R² = {result['mean_r2']:.6f} (+/- {result['std_r2']:.6f})")
    
    # Sort results by mean R² descending
    results_sorted = sorted(results, key=lambda x: x['mean_r2'], reverse=True)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            'Model': r['name'],
            'Mean R²': r['mean_r2'],
            'Std R²': r['std_r2'],
            'CV Scores': r['cv_scores']
        }
        for r in results_sorted
    ])
    
    return comparison_df, results_sorted

def analyze_feature_importance(model, X, feature_names):
    """
    Extract feature importance from tree-based model
    Returns sorted DataFrame
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df

# ============== RESULT LOGGING & PERSISTENCE ==============

def ensure_log_dir(log_dir='./results_logs'):
    """Create logs directory if it doesn't exist"""
    Path(log_dir).mkdir(exist_ok=True)
    return log_dir

def save_cv_results(comparison_df, results, log_dir='./results_logs'):
    """
    Save cross-validation results to JSON and CSV files
    
    Args:
        comparison_df: DataFrame with model comparison results
        results: List of result dictionaries from compare_models()
        log_dir: Directory to save logs
    """
    log_dir = ensure_log_dir(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save comparison table as CSV
    csv_path = os.path.join(log_dir, f'model_comparison_{timestamp}.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved model comparison to: {csv_path}")
    
    # Save detailed results as JSON
    json_path = os.path.join(log_dir, f'cv_results_{timestamp}.json')
    json_data = {
        'timestamp': timestamp,
        'models': []
    }
    
    for r in results:
        json_data['models'].append({
            'name': r['name'],
            'mean_r2': float(r['mean_r2']),
            'std_r2': float(r['std_r2']),
            'cv_scores': [float(x) for x in r['cv_scores']]
        })
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved detailed CV results to: {json_path}")
    
    return csv_path, json_path

def save_feature_importance(importance_df, model_name, log_dir='./results_logs'):
    """
    Save feature importance results to CSV
    
    Args:
        importance_df: DataFrame with feature importances
        model_name: Name of the model
        log_dir: Directory to save logs
    """
    log_dir = ensure_log_dir(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    csv_path = os.path.join(log_dir, f'feature_importance_{model_name}_{timestamp}.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved feature importance to: {csv_path}")
    
    return csv_path

def load_cv_results(json_path):
    """
    Load cross-validation results from JSON file
    
    Args:
        json_path: Path to JSON results file
        
    Returns:
        DataFrame with model comparison results
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    models_list = []
    for model in data['models']:
        models_list.append({
            'Model': model['name'],
            'Mean R²': model['mean_r2'],
            'Std R²': model['std_r2'],
            'CV Scores': model['cv_scores']
        })
    
    return pd.DataFrame(models_list)

def load_feature_importance(csv_path):
    """
    Load feature importance results from CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with feature importances
    """
    return pd.read_csv(csv_path)

def list_available_logs(log_dir='./results_logs'):
    """
    List all available log files in the logs directory
    
    Args:
        log_dir: Directory containing logs
        
    Returns:
        Dictionary with categorized log files
    """
    if not os.path.exists(log_dir):
        print(f"No log directory found at {log_dir}")
        return {}
    
    files = os.listdir(log_dir)
    
    logs = {
        'cv_results': [f for f in files if f.startswith('cv_results_')],
        'model_comparison': [f for f in files if f.startswith('model_comparison_')],
        'feature_importance': [f for f in files if f.startswith('feature_importance_')]
    }
    
    print("Available log files:")
    print("-" * 50)
    for category, file_list in logs.items():
        if file_list:
            print(f"\n{category}:")
            for f in sorted(file_list, reverse=True):
                print(f"  - {f}")
    
    return logs

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
    
    # For AdaBoostRegressor: wrap a DT with max_depth
    elif class_name == 'AdaBoostRegressor':
        base_estimator = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
        return model_class(estimator=base_estimator, n_estimators=100, learning_rate=0.1, random_state=random_state)
    
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

def compute_and_save_complexity_analysis_multi(models_list, X, y, depths=range(1, 16), k=5, log_dir='./results_logs'):
    """
    Compute bias-variance tradeoff for multiple models by varying complexity (max_depth)
    Saves results to CSV with model names for comparison.
    **INCLUDES CHECKPOINTING**: Saves after each model completes to enable resuming from crashes.
    
    Args:
        models_list: List of tuples (model_name, model_instance)
        X: Feature matrix
        y: Target values
        depths: Range of complexity values to test
        k: Number of CV folds
        log_dir: Directory to save logs
    """
    log_dir = ensure_log_dir(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("\nComputing complexity analysis for top 3 models (bias-variance tradeoff)...")
    print("-" * 70)
    
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
                
                # Get CV scores
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
                
                print(f"  Depth {depth:2d}: Train Error = {train_error:.6f}, Val Error = {val_error:.6f}")
                
            except Exception as e:
                print(f"  Depth {depth:2d}: ERROR - {str(e)}")
                continue
        
        # **CHECKPOINT**: Save after each model completes
        if model_results:
            df_checkpoint = pd.DataFrame(all_results)
            df_checkpoint.to_csv(checkpoint_path, index=False)
            print(f"  ✓ Checkpoint saved: {len(all_results)} results so far")
    
    # Final save with proper naming
    final_csv_path = os.path.join(log_dir, f'complexity_analysis_top3_{timestamp}.csv')
    df = pd.DataFrame(all_results)
    df.to_csv(final_csv_path, index=False)
    print(f"\n✓ Saved complexity analysis for top 3 models to: {final_csv_path}")
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return final_csv_path

# ============== 3D MODELING - OVERFITTING/UNDERFITTING ANALYSIS ==============

def plot_hyperparameter_surface(hyperparams_grid=None, train_errors=None, val_errors=None, param_names=None, save_path=None, log_file=None):
    """
    Visualize training vs validation errors as 2D plot with 2 lines.
    
    Can be called in two ways:
    1. With live data: pass hyperparams_grid, train_errors, val_errors, param_names
    2. With log file: pass log_file path (loads from saved results)
    
    Args:
        hyperparams_grid: Dict with param names as keys and lists of values
        train_errors: Array or 2D array of training errors
        val_errors: Array or 2D array of validation errors
        param_names: Tuple of (x_param_name, y_param_name) - optional
        save_path: Optional path to save figure
        log_file: Optional path to CSV log file with results (for reuse)
    
    Shows training vs validation errors as 2 colored lines for comparison.
    """
    if log_file:
        print(f"Loading hyperparameter data from {log_file}...")
        pass
    
    if any(x is None for x in [train_errors, val_errors]):
        raise ValueError("Must provide train_errors and val_errors")
    
    # Flatten arrays if 2D
    train_errors_flat = np.array(train_errors).flatten()
    val_errors_flat = np.array(val_errors).flatten()
    x_axis = np.arange(len(train_errors_flat))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot training and validation errors as 2 lines with different colors
    ax.plot(x_axis, train_errors_flat, 'b-', linewidth=2.5, label='Training Error', marker='o', markersize=6, alpha=0.8)
    ax.plot(x_axis, val_errors_flat, 'r-', linewidth=2.5, label='Validation Error', marker='s', markersize=6, alpha=0.8)
    
    # Shade the gap between curves to highlight overfitting
    ax.fill_between(x_axis, train_errors_flat, val_errors_flat, alpha=0.15, color='gray', label='Generalization Gap')
    
    ax.set_xlabel('Hyperparameter Configuration Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Validation Error - Hyperparameter Analysis', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curve(data_fractions=None, train_errors=None, val_errors=None, save_path=None, log_file=None):
    """
    Visualize learning curves with training vs validation errors as 2D plot.
    
    Can be called in two ways:
    1. With live data: pass data_fractions, train_errors, val_errors
    2. With log file: pass log_file path (loads from saved results)
    
    Args:
        data_fractions: Array of training set fractions (0.1 to 1.0)
        train_errors: Array of training errors for each fraction
        val_errors: Array of validation errors for each fraction
        save_path: Optional path to save figure
        log_file: Optional path to CSV log file with results (for reuse)
    
    Shows if model needs more data (converging curves) or has systematic bias (parallel high curves).
    """
    if log_file:
        print(f"Loading learning curve data from {log_file}...")
        df = pd.read_csv(log_file)
        data_fractions = df['data_fraction'].values
        train_errors = df['train_error'].values
        val_errors = df['val_error'].values
    
    if any(x is None for x in [data_fractions, train_errors, val_errors]):
        raise ValueError("Must provide either log_file or all of: data_fractions, train_errors, val_errors")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot training and validation error lines with different colors
    ax.plot(data_fractions, train_errors, 'b-', linewidth=2.5, label='Training Error', marker='o', markersize=8, alpha=0.8)
    ax.plot(data_fractions, val_errors, 'r-', linewidth=2.5, label='Validation Error', marker='s', markersize=8, alpha=0.8)
    
    # Shade the gap between curves to show bias-variance
    ax.fill_between(data_fractions, train_errors, val_errors, alpha=0.15, color='gray', label='Bias-Variance Gap')
    
    ax.set_xlabel('Training Data Fraction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curves - Bias-Variance Analysis', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_complexity_bias_variance(complexities=None, train_errors=None, val_errors=None, save_path=None, log_file=None):
    """
    Visualize model complexity vs training/validation errors as 2D plot.
    
    Can be called in two ways:
    1. With live data: pass complexities, train_errors, val_errors
    2. With log file: pass log_file path (loads from saved results)
    
    Args:
        complexities: Array of complexity values (e.g., max_depth 1-15)
        train_errors: Array of training errors for each complexity level
        val_errors: Array of validation errors for each complexity level
        save_path: Optional path to save figure
        log_file: Optional path to CSV log file with results (for reuse)
    
    Directly visualizes training vs validation errors as 2 colored lines showing bias-variance tradeoff.
    """
    if log_file:
        print(f"Loading complexity analysis data from {log_file}...")
        df = pd.read_csv(log_file)
        complexities = df['complexity'].values
        train_errors = df['train_error'].values
        val_errors = df['val_error'].values
    
    if any(x is None for x in [complexities, train_errors, val_errors]):
        raise ValueError("Must provide either log_file or all of: complexities, train_errors, val_errors")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot both curves on 2D as colored lines
    ax.plot(complexities, train_errors, 'b-', linewidth=2.5, 
           label='Training Error', marker='o', markersize=7, alpha=0.8)
    ax.plot(complexities, val_errors, 'r-', linewidth=2.5, 
           label='Validation Error', marker='s', markersize=7, alpha=0.8)
    
    # Find optimal complexity (minimum validation error)
    optimal_idx = np.argmin(val_errors)
    ax.scatter([complexities[optimal_idx]], [val_errors[optimal_idx]], 
              color='gold', s=300, marker='*', 
              label=f'Optimal (depth={int(complexities[optimal_idx])})', zorder=10, edgecolors='black', linewidth=2)
    
    # Shade the gap between curves to show bias-variance gap
    ax.fill_between(complexities, train_errors, val_errors, alpha=0.15, color='gray', label='Bias-Variance Gap')
    
    # Add shaded regions for underfitting and overfitting
    ax.axvspan(complexities[0], complexities[optimal_idx], alpha=0.1, color='red', label='Underfitting Region')
    ax.axvspan(complexities[optimal_idx], complexities[-1], alpha=0.1, color='orange', label='Overfitting Region')
    
    ax.set_xlabel('Model Complexity (Max Depth)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('Bias-Variance Tradeoff - Model Complexity Analysis', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============== MAIN EXECUTION ==============

if __name__ == "__main__":
    
    print("="*70)
    print("TAXI DURATION PREDICTION - MODEL OPTIMIZATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('Train.csv')
    
    # Sample 30% for faster iteration
    taxi = df.sample(frac=0.3, random_state=42).copy()
    print(f"Using 30% of training data: {len(taxi)} rows (out of {len(df)} total)")
    
    # Feature engineering
    print("\nApplying feature engineering...")
    taxi = engineer_features(taxi)
    
    # Define feature set
    feature_columns = [
        # Original features
        'start_hour', 'start_minute', 'start_day', 'start_month', 
        'x2xDistance', 'dayOfWeek', 
        'additionalStop', 'is_weekend', 
        # Polynomial features
        'x2xDistance_squared', 'y2yDistance_squared', 
        'x2xDistance_cubed', 'y2yDistance_cubed',
        # Interaction terms
        'distance_x_timeOfDay', 'distance_y_timeOfDay', 
        'distance_x_weekend', 'distance_y_weekend',
        'distance_x_hour', 'distance_y_hour',
        # Cyclic encodings
        'hour_sin', 'hour_cos', 'month_sin', 
        'dayOfWeek_sin', 'dayOfWeek_cos',
        # Distance metrics
        'manhattan_distance_squared', 'total_distance',
        'distance_category',
        # Destination interactions
        'distance_x_dest_x', 'distance_y_dest_y',
        # Time interactions
        'hour_x_is_weekend', 'timeOfDay_x_season'
    ]
    
    X = taxi[feature_columns]
    y = taxi['duration']
    
    print(f"Features: {len(feature_columns)} | Samples: {len(X)}")
    
    # Define models - TREE-BASED METHODS with BOOSTING/BAGGING
    print("\n" + "="*70)
    print("INITIALIZING MODELS (with GPU support where available)")
    print("="*70)
    
    models = {
        # Bagging methods
        'Random Forest': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
        ),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
        ),
    }
    
    # Add XGBoost if available
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            random_state=42, n_jobs=-1, verbosity=0
        )
    
    # Evaluate all models with 5-fold CV
    print(f"\nRunning 5-fold Cross-Validation on {len(models)} models...")
    print("-"*70)
    
    comparison_df, results = compare_models(models, X, y, k=5)
    
    # Display results
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS (Sorted by Mean R²)")
    print("="*70)
    print(comparison_df.to_string())
    
    # Save CV results
    csv_path, json_path = save_cv_results(comparison_df, results, log_dir='./results_logs')
    
    # Best model
    best_result = results[0]
    best_model = best_result['model']
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_result['name']}")
    print(f"Mean R² Score: {best_result['mean_r2']:.6f}")
    print(f"Std Dev: {best_result['std_r2']:.6f}")
    print(f"{'='*70}")
    
    # Compute complexity analysis for top 3 models
    top_3_models = [(r['name'], type(r['model'])) for r in results[:3]]
    complexity_csv = compute_and_save_complexity_analysis_multi(top_3_models, X, y, depths=range(1, 16), log_dir='./results_logs')
    
    # Feature importance analysis for best model
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nFEATURE IMPORTANCE for {best_result['name']}:")
        print("-"*70)
        
        # Fit model on full data to get importances
        best_model.fit(X, y)
        importance_df = analyze_feature_importance(best_model, X, feature_columns)
        
        print(importance_df.head(15).to_string())
        
        # Show features with low importance (candidates for removal)
        print("\n\nLOW IMPORTANCE FEATURES (candidates for removal):")
        print(importance_df.tail(10).to_string())
        
        # Summary
        print(f"\nTotal features: {len(importance_df)}")
        print(f"Top 10 features account for {importance_df.head(10)['Importance'].sum():.2%} of importance")
        
        # Save feature importance
        imp_csv_path = save_feature_importance(importance_df, best_result['name'], log_dir='./results_logs')
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("1. Results have been saved to ./results_logs/")
    print("2. Use plot_complexity_bias_variance() with log files to replot without rerunning")
    print("3. Use list_available_logs() to see all saved results")
    print("4. Remove low-importance features and re-evaluate")
    print("5. Tune hyperparameters of best model")
    print("6. Try additional ensemble methods or stacking")
    print("="*70)

