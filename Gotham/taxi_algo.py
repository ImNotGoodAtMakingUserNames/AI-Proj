# Imports
from enum import IntEnum
from dataclasses import dataclass
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Neural Network imports - using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Gradient Boosting and Optimization imports
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error

from ISLP.models import (ModelSpec as MS, summarize)


# ============== MODEL TOGGLES ==============
# Set to True to run a model, False to skip it
ENABLE_LINEAR_REGRESSION = False
ENABLE_RIDGE_REGRESSION = False
ENABLE_LASSO_REGRESSION = True
ENABLE_RANDOM_FOREST = False
ENABLE_GRADIENT_BOOSTING = False
ENABLE_BAGGING_REGRESSOR = False
ENABLE_NEURAL_NETWORK = False
ENABLE_XGBOOST_OPTUNA = False
ENABLE_LIGHTGBM_OPTUNA = False
ENABLE_EXTRATREES_OPTUNA = False          # Disabled: froze at trial 85 with 0.78
ENABLE_CATBOOST_OPTUNA = False
ENABLE_RANDOMFOREST_OPTUNA = False        # Disabled: too slow (27h est), underperforming (~0.777 vs LGB 0.826)
ENABLE_HISTGB_OPTUNA = False              # Disabled: prioritize faster models
ENABLE_WEIGHTED_ENSEMBLE = False
ENABLE_STACKING = False
ENABLE_FEATURE_IMPORTANCE = False
# ============================================


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
    return int(day_of_week >= 5)


def getTimeofDay(start_hour):
    if start_hour < 8:
        return 0 # Early morning
    if start_hour >= 8 and start_hour < 11:
        return 1 # Morning
    if start_hour >= 11 and start_hour < 13:
        return 2 # Lunch
    if start_hour >= 13 and start_hour < 16:
        return 3 # Afternoon
    if start_hour >= 16 and start_hour < 18:
        return 4 # Evening
    if start_hour >= 18 and start_hour < 24:
        return 5 # Night


def isHoliday(start_month, start_day, start_year):
    """Check if the date is a US federal holiday"""
    if any(pd.isna(x) for x in [start_month, start_day, start_year]):
        raise ValueError(f"One or more date components are NaN: month={start_month}, day={start_day}, year={start_year}")
    
    month = int(start_month)
    day = int(start_day)
    year = int(start_year)
    
    holidays_2034 = {
        (1, 1),    # New Year's Day
        (1, 16),   # MLK Jr Birthday
        (2, 20),   # Presidents Day
        (5, 29),   # Memorial Day
        (6, 19),   # Juneteenth
        (7, 4),    # Independence Day
        (9, 4),    # Labor Day
        (10, 9),   # Columbus Day
        (11, 11),  # Veterans Day
        (11, 23),  # Thanksgiving Day
        (12, 25),  # Christmas Day
    }
    
    if year != 2034:
        raise ValueError(f"Holiday checking is only configured for year 2034, got {year}")
    
    return int((month, day) in holidays_2034)

def evaluate_model_kfold(X, y, k=5):
    """
    Perform k-fold cross validation on a linear regression model.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix (without constant)
    y : Series
        Target variable
    k : int, default=5
        Number of folds for cross-validation
    
    Returns:
    --------
    dict : Dictionary containing cross-validation results
        - 'cv_scores': array of R² scores for each fold
        - 'mean_cv_score': mean R² score across all folds
        - 'std_cv_score': standard deviation of R² scores
        - 'model': fitted LinearRegression model on full data (for reference)
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    model = LinearRegression()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
    
    # Fit model on full data for reference
    model.fit(X, y)
    
    results = {
        'cv_scores': cv_scores,
        'mean_cv_score': cv_scores.mean(),
        'std_cv_score': cv_scores.std(),
        'model': model
    }
    
    return results

# Read Data
df = pd.read_csv('Train.csv')
# Sample 30% of data for exploration
taxi = df.sample(frac=0.3, random_state=42).copy()

print('==============|Training Data Imported|==============')
print(f'Using 30% of training data ({len(taxi)} rows out of {len(df)} total rows)\n')

# Extract granular datetime components from pickup_datetime
taxi['start_hour'] = taxi['pickup_datetime'].apply(getStartHour)
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

taxi['end_hour'] = taxi['end_time'].apply(getEndHour)
taxi['end_minute'] = taxi['end_time'].apply(getEndMinute)
taxi['end_second'] = taxi['end_time'].apply(getEndSecond)


taxi['timeOfDay'] = taxi.apply(lambda row: getTimeofDay(row['start_hour']), axis=1)
# print(taxi['timeOfDay'])

taxi['is_weekend'] = taxi['dayOfWeek'].apply(isWeekend)
# print(taxi['is_weekend'])

taxi['additionalStop'] = (taxi['NumberOfPassengers'] > 1).astype(int)
# print(taxi['additionalStop'])

taxi['is_holiday'] = taxi.apply(lambda row: isHoliday(row['start_month'], row['start_day'], row['start_year']), axis=1)
# print(taxi['is_holiday'])

# ============== FEATURE ENGINEERING ==============
print('==============|Beginning Feature Engineering|==============\n')

# 1. Polynomial features for distance (capture non-linear distance decay)
taxi['x2xDistance_squared'] = taxi['x2xDistance'] ** 2
taxi['y2yDistance_squared'] = taxi['y2yDistance'] ** 2
taxi['x2xDistance_cubed'] = taxi['x2xDistance'] ** 3
taxi['y2yDistance_cubed'] = taxi['y2yDistance'] ** 3

# 2. Interaction terms (taxi behavior varies by context)
taxi['distance_x_timeOfDay'] = taxi['x2xDistance'] * taxi['timeOfDay']
taxi['distance_y_timeOfDay'] = taxi['y2yDistance'] * taxi['timeOfDay']
taxi['distance_x_weekend'] = taxi['x2xDistance'] * taxi['is_weekend']
taxi['distance_y_weekend'] = taxi['y2yDistance'] * taxi['is_weekend']
taxi['distance_x_hour'] = taxi['x2xDistance'] * taxi['start_hour']
taxi['distance_y_hour'] = taxi['y2yDistance'] * taxi['start_hour']

# 3. Cyclic encodings for periodic features (hour and month are cyclic, not linear)
taxi['hour_sin'] = np.sin(2 * np.pi * taxi['start_hour'] / 24)
taxi['hour_cos'] = np.cos(2 * np.pi * taxi['start_hour'] / 24)
taxi['month_sin'] = np.sin(2 * np.pi * taxi['start_month'] / 12)
taxi['month_cos'] = np.cos(2 * np.pi * taxi['start_month'] / 12)
taxi['dayOfWeek_sin'] = np.sin(2 * np.pi * taxi['dayOfWeek'] / 7)
taxi['dayOfWeek_cos'] = np.cos(2 * np.pi * taxi['dayOfWeek'] / 7)

# 4. Manhattan distance (alternative to Euclidean)
taxi['manhattan_distance'] = taxi['x2xDistance'] + taxi['y2yDistance']
taxi['manhattan_distance_squared'] = taxi['manhattan_distance'] ** 2

# 5. Total distance metric
taxi['total_distance'] = np.sqrt(taxi['x2xDistance'] ** 2 + taxi['y2yDistance'] ** 2)

# 6. Distance bins/categories (capture non-linear pricing tiers)
taxi['distance_category'] = pd.cut(taxi['total_distance'], bins=[0, 1, 2, 5, 10, float('inf')], labels=['very_short', 'short', 'medium', 'long', 'very_long']).cat.codes

# 7. Interaction: distance × destination location (origin-destination patterns)
taxi['distance_x_dest_x'] = taxi['x2xDistance'] * taxi['dropoff_x']
taxi['distance_y_dest_y'] = taxi['y2yDistance'] * taxi['dropoff_y']

# 8. Time-based interactions
taxi['hour_x_is_weekend'] = taxi['start_hour'] * taxi['is_weekend']
taxi['timeOfDay_x_season'] = taxi['timeOfDay'] * taxi['season']

print('==============|Training Data Filtered/Modified|==============\n')
# print('==============|Beginning Linear Regression 1|==============\n')

# # Linear Regression - All features
# features1 = ['start_hour', 'start_minute', 'start_second', 'start_day', 'start_month', 'start_year', 'x2xDistance', 'y2yDistance', 'dayOfWeek',
#                 'season', 'end_hour', 'end_minute', 'end_second', 'timeOfDay', 'additionalStop', 'is_weekend', 'is_holiday']

# x1 = taxi[features1]
# x1 = sm.add_constant(x1)

# y1 = taxi['duration']

# model1 = sm.OLS(y1, x1)
# results1 = model1.fit()

# print(results1.summary())


# print('==============|Linear Regression 1 Complete|==============\n')
# First run had adjusted R squared value of 0.598
# First run shows the following with p values higher than 0.05: start_second, end_minute, end_second, season, y2yDistance, timeOfDay, month_cos, manhattan_distance
# Removed after Forest analysis: 'is_holiday',

# ============== PREPARE FEATURES FOR ALL MODELS ==============
# Define feature set and prepare data that all models will use
# NOTE: Excluding end_hour, end_minute, end_second as they depend on duration (our target)
# This prevents data leakage where features derived from the target variable would give unfair advantage
features2 = [
    # Original features (start only, not end - end depends on duration)
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

print(f"Total engineered features: {len(features2)}")
print(f"Feature categories: 12 original + 4 polynomial + 6 interaction + 6 cyclic + 4 distance + 2 destination + 2 time = {len(features2)} total\n")

x2_no_const = taxi[features2]  # sklearn doesn't need manual constant addition

# Scale features for Ridge/Lasso and other models
scaler = StandardScaler()
x2_scaled = scaler.fit_transform(x2_no_const)

y2 = taxi['duration']
# ============================================================

# Removing >0.05 p values:
if ENABLE_LINEAR_REGRESSION:
    print('==============|Beginning Linear Regression 2|==============\n')

    x2 = x2_no_const.copy()
    x2 = sm.add_constant(x2)

    model2 = sm.OLS(y2, x2)
    results2 = model2.fit()

    print(results2.summary())

    print('==============|Linear Regression 2 Complete|==============\n')

    # K-Fold Cross Validation for Model 2 (Linear - baseline)
    print('==============|K-Fold Cross Validation - Model 2 (Linear)|==============\n')

    # Linear Regression (unscaled for comparison)
    linear_model = LinearRegression()
    linear_cv_scores = cross_val_score(linear_model, x2_scaled, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')

    print(f"Linear Regression CV R² Scores: {linear_cv_scores}")
    print(f"Linear Mean CV R²: {linear_cv_scores.mean():.6f}")
    print(f"Linear Std Dev CV R²: {linear_cv_scores.std():.6f}")
    print('==============|Linear Regression CV Complete|==============\n')
else:
    print('SKIPPED: Linear Regression (ENABLE_LINEAR_REGRESSION = False)\n')
    linear_cv_scores = None
    results2 = None


# Ridge Regression with GridSearchCV
if ENABLE_RIDGE_REGRESSION:
    print('==============|Ridge Regression - Model 3|==============\n')
    ridge_alphas = np.logspace(-3, 4, 150)  # Expanded range: 0.001 to 10000
    ridge_model = Ridge()

    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    ridge_grid = GridSearchCV(ridge_pipeline, {'ridge__alpha': ridge_alphas}, cv=5, scoring='r2', n_jobs=12)
    ridge_grid.fit(x2_no_const, y2)

    print(f"Best Ridge Alpha: {ridge_grid.best_params_['ridge__alpha']:.6f}")
    print(f"Best Ridge CV R² Score: {ridge_grid.best_score_:.6f}")

    # Evaluate best Ridge model with k-fold
    ridge_best = Ridge(alpha=ridge_grid.best_params_['ridge__alpha'])
    ridge_cv_scores = cross_val_score(ridge_best, x2_scaled, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"Ridge CV R² Scores (5 folds): {ridge_cv_scores}")
    print(f"Ridge Mean CV R²: {ridge_cv_scores.mean():.6f}")
    print(f"Ridge Std Dev CV R²: {ridge_cv_scores.std():.6f}")

    # Show top 10 alpha values tested
    top_alphas_ridge = sorted(zip(ridge_grid.cv_results_['param_ridge__alpha'], ridge_grid.cv_results_['mean_test_score']), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 Ridge Alpha values tested:")
    for alpha, score in top_alphas_ridge:
        print(f"  Alpha: {alpha:.6f} -> R²: {score:.6f}")
    print('==============|Ridge Regression Complete|==============\n')
else:
    print('SKIPPED: Ridge Regression (ENABLE_RIDGE_REGRESSION = False)\n')
    ridge_cv_scores = None


# Lasso Regression with GridSearchCV
lasso_cv_scores = None
if ENABLE_LASSO_REGRESSION:
    print('==============|Lasso Regression - Model 4|==============\n')
    print('Note: Lasso may show ConvergenceWarning - this is normal with many features. Increasing iterations to handle engineered features...\n')
    lasso_alphas = np.logspace(-4, 1, 150)  # Expanded range: 0.0001 to 10
    lasso_model = Lasso(max_iter=200000, tol=1e-3, warm_start=False)

    lasso_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(max_iter=200000, tol=1e-3, warm_start=False))
    ])

    lasso_grid = GridSearchCV(lasso_pipeline, {'lasso__alpha': lasso_alphas}, cv=5, scoring='r2', n_jobs=12)
    lasso_grid.fit(x2_no_const, y2)

    print(f"Best Lasso Alpha: {lasso_grid.best_params_['lasso__alpha']:.6f}")
    print(f"Best Lasso CV R² Score: {lasso_grid.best_score_:.6f}")

    # Evaluate best Lasso model with k-fold
    lasso_best = Lasso(alpha=lasso_grid.best_params_['lasso__alpha'], max_iter=200000, tol=1e-3, warm_start=False)
    lasso_cv_scores = cross_val_score(lasso_best, x2_scaled, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"Lasso CV R² Scores (5 folds): {lasso_cv_scores}")
    print(f"Lasso Mean CV R²: {lasso_cv_scores.mean():.6f}")
    print(f"Lasso Std Dev CV R²: {lasso_cv_scores.std():.6f}")

    # Show top 10 alpha values tested
    top_alphas_lasso = sorted(zip(lasso_grid.cv_results_['param_lasso__alpha'], lasso_grid.cv_results_['mean_test_score']), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 Lasso Alpha values tested:")
    for alpha, score in top_alphas_lasso:
        print(f"  Alpha: {alpha:.6f} -> R²: {score:.6f}")
    print('==============|Lasso Regression Complete|==============\n')


# Model Comparison Summary
print('==============|Model Comparison Summary|==============')
if ENABLE_LINEAR_REGRESSION and linear_cv_scores is not None:
    print(f"Linear Regression (Model 2)     - Mean CV R²: {linear_cv_scores.mean():.6f} ± {linear_cv_scores.std():.6f}")
if ENABLE_RIDGE_REGRESSION and ridge_cv_scores is not None:
    print(f"Ridge Regression (Model 3)      - Mean CV R²: {ridge_cv_scores.mean():.6f} ± {ridge_cv_scores.std():.6f}")
if ENABLE_LASSO_REGRESSION and lasso_cv_scores is not None:
    print(f"Lasso Regression (Model 4)      - Mean CV R²: {lasso_cv_scores.mean():.6f} ± {lasso_cv_scores.std():.6f}")
print('==============|Comparison Complete|==============\n')

# Random Forest Regression with minimal parameter search (FAST)
if ENABLE_RANDOM_FOREST:
    print('==============|Random Forest Regression - Model 5|==============\n')

    # Minimal params for speed on 300K rows: only 12 combinations with 3-fold CV
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf_model = RandomForestRegressor(random_state=42, n_jobs=12)
    rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='r2', n_jobs=12, verbose=1)

    print("Training Random Forest with GridSearchCV (12 combinations, 5-fold CV)...\n")
    rf_grid.fit(x2_no_const, y2)

    print(f"\nBest Random Forest Hyperparameters:")
    for param, value in rf_grid.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest Random Forest CV R² Score: {rf_grid.best_score_:.6f}")

    # Evaluate best Random Forest model with k-fold (5-fold for final evaluation)
    rf_best = RandomForestRegressor(
        n_estimators=rf_grid.best_params_['n_estimators'],
        max_depth=rf_grid.best_params_['max_depth'],
        min_samples_split=rf_grid.best_params_['min_samples_split'],
        min_samples_leaf=rf_grid.best_params_['min_samples_leaf'],
        max_features=rf_grid.best_params_['max_features'],
        random_state=42,
        n_jobs=12
    )

    rf_cv_scores = cross_val_score(rf_best, x2_no_const, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=12)
    print(f"\nRandom Forest CV R² Scores (5 folds): {rf_cv_scores}")
    print(f"Random Forest Mean CV R²: {rf_cv_scores.mean():.6f}")
    print(f"Random Forest Std Dev CV R²: {rf_cv_scores.std():.6f}")

    # Show top 10 hyperparameter combinations tested
    top_rf_combos = sorted(zip(rf_grid.cv_results_['params'], rf_grid.cv_results_['mean_test_score']), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop Random Forest hyperparameter combinations:")
    for idx, (params, score) in enumerate(top_rf_combos, 1):
        print(f"  {idx}. n_est={params['n_estimators']}, depth={params['max_depth']}, split={params['min_samples_split']}, leaf={params['min_samples_leaf']}, max_feat={params['max_features']} -> R²: {score:.6f}")

    print('==============|Random Forest Regression Complete|==============\n')
else:
    print('SKIPPED: Random Forest Regression (ENABLE_RANDOM_FOREST = False)\n')
    rf_cv_scores = None
    rf_final = None


# Gradient Boosting Regression (FAST VERSION)
if ENABLE_GRADIENT_BOOSTING:
    print('==============|Gradient Boosting Regression - Model 6|==============\n')

    # Optimized for speed: RandomizedSearchCV with 12 iterations and 2-fold CV
    # Note: colsample_bytree is XGBoost-only, not available in sklearn's GradientBoostingRegressor
    gb_param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 4, 5],
        'subsample': [0.6, 0.8, 1.0],
        'min_samples_split': [2, 5, 10]
    }

    gb_model = GradientBoostingRegressor(random_state=42)
    gb_grid = RandomizedSearchCV(gb_model, gb_param_grid, n_iter=12, cv=5, scoring='r2', n_jobs=12, verbose=1, random_state=42)

    print("Training Gradient Boosting with RandomizedSearchCV (12 random iterations, 5-fold CV)...\n")
    gb_grid.fit(x2_no_const, y2)

    print(f"\nBest Gradient Boosting Hyperparameters:")
    for param, value in gb_grid.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest Gradient Boosting CV R² Score: {gb_grid.best_score_:.6f}")

    # Evaluate best GB model with k-fold (5-fold for final evaluation)
    gb_best = GradientBoostingRegressor(
        n_estimators=gb_grid.best_params_['n_estimators'],
        learning_rate=gb_grid.best_params_['learning_rate'],
        max_depth=gb_grid.best_params_['max_depth'],
        subsample=gb_grid.best_params_['subsample'],
        min_samples_split=gb_grid.best_params_['min_samples_split'],
        random_state=42
    )

    gb_cv_scores = cross_val_score(gb_best, x2_no_const, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"\nGradient Boosting CV R² Scores (5 folds): {gb_cv_scores}")
    print(f"Gradient Boosting Mean CV R²: {gb_cv_scores.mean():.6f}")
    print(f"Gradient Boosting Std Dev CV R²: {gb_cv_scores.std():.6f}")

    # Show top hyperparameter combinations tested
    top_gb_combos = sorted(zip(gb_grid.cv_results_['params'], gb_grid.cv_results_['mean_test_score']), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop Gradient Boosting hyperparameter combinations:")
    for idx, (params, score) in enumerate(top_gb_combos, 1):
        print(f"  {idx}. n_est={params['n_estimators']}, lr={params['learning_rate']}, depth={params['max_depth']}, subsample={params['subsample']}, min_split={params['min_samples_split']} -> R²: {score:.6f}")

    print('==============|Gradient Boosting Regression Complete|==============\n')
else:
    print('SKIPPED: Gradient Boosting Regression (ENABLE_GRADIENT_BOOSTING = False)\n')
    gb_cv_scores = None


# Bagging Regressor with Decision Trees
if ENABLE_BAGGING_REGRESSOR:
    print('==============|Bagging Regressor (with Decision Trees) - Model 7|==============\n')

    # Bagging with Decision Trees
    bagging_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_samples': [0.6, 0.8, 1.0],
        'max_features': [0.6, 0.8, 1.0],
        'bootstrap': [True]
    }

    bagging_model = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        random_state=42,
        n_jobs=12
    )

    bagging_grid = GridSearchCV(bagging_model, bagging_param_grid, cv=5, scoring='r2', n_jobs=12, verbose=1)

    print("Training Bagging Regressor with GridSearchCV...\n")
    bagging_grid.fit(x2_no_const, y2)

    print(f"\nBest Bagging Hyperparameters:")
    for param, value in bagging_grid.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest Bagging CV R² Score: {bagging_grid.best_score_:.6f}")

    # Evaluate best Bagging model with k-fold
    bagging_best = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=bagging_grid.best_params_['n_estimators'],
        max_samples=bagging_grid.best_params_['max_samples'],
        max_features=bagging_grid.best_params_['max_features'],
        bootstrap=bagging_grid.best_params_['bootstrap'],
        random_state=42,
        n_jobs=12
    )

    bagging_cv_scores = cross_val_score(bagging_best, x2_no_const, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=12)
    print(f"\nBagging CV R² Scores (5 folds): {bagging_cv_scores}")
    print(f"Bagging Mean CV R²: {bagging_cv_scores.mean():.6f}")
    print(f"Bagging Std Dev CV R²: {bagging_cv_scores.std():.6f}")

    # Show top hyperparameter combinations
    top_bagging_combos = sorted(zip(bagging_grid.cv_results_['params'], bagging_grid.cv_results_['mean_test_score']), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop Bagging hyperparameter combinations:")
    for idx, (params, score) in enumerate(top_bagging_combos, 1):
        print(f"  {idx}. n_est={params['n_estimators']}, max_samples={params['max_samples']}, max_features={params['max_features']} -> R²: {score:.6f}")

    print('==============|Bagging Regressor Complete|==============\n')
else:
    print('SKIPPED: Bagging Regressor (ENABLE_BAGGING_REGRESSOR = False)\n')
    bagging_cv_scores = None


# Neural Network Regression with Hyperparameter Tuning (FAST SCREENING)
if ENABLE_NEURAL_NETWORK:
    print('==============|Neural Network - Fast Scaler + Config Screening|==============\n')
    print("Quick evaluation of scaler strategies and architectures (20-30 min runtime)\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ============ LIGHTWEIGHT HYPERPARAMETER CONFIGS (just 3 to test quickly) ============
    hyperparameter_configs = [
        ([128, 64, 32], 0.001, [0.1, 0.1, 0.1], 32, 15),  # Balanced (original)
        ([256, 128, 64], 0.001, [0.1, 0.1], 32, 15),      # Wider, fewer layers
        ([64, 32], 0.001, [0.1], 32, 15),                 # Simpler
    ]

    scaler_configs = [
        ('StandardScaler', StandardScaler()),
        ('MinMaxScaler', MinMaxScaler()),
        ('RobustScaler', RobustScaler()),
        ('NoScaling', None)
    ]

    def train_nn_model_fast(X_train_scaled, y_train, X_val_scaled, y_val, hidden_layers, learning_rate, 
                            dropout_rates, batch_size, early_stop_patience, device):
        """Lightweight NN training (80 epochs max, early stopping at 10 patience)"""
        class ConfigurableNN(nn.Module):
            def __init__(self, input_size, hidden_layers, dropout_rates):
                super(ConfigurableNN, self).__init__()
                
                layer_sizes = [input_size] + hidden_layers + [1]
                self.layers = nn.ModuleList()
                self.batch_norms = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                
                for i in range(len(layer_sizes) - 2):
                    self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                    self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                    dropout_rate = dropout_rates[i] if i < len(dropout_rates) else 0.1
                    self.dropouts.append(nn.Dropout(dropout_rate))
                
                self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
            
            def forward(self, x):
                for i in range(len(self.layers) - 1):
                    x = self.layers[i](x)
                    x = self.batch_norms[i](x)
                    x = torch.relu(x)
                    x = self.dropouts[i](x)
                x = self.layers[-1](x)
                return x
        
        model = ConfigurableNN(X_train_scaled.shape[1], hidden_layers, dropout_rates).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                         patience=5, min_lr=1e-6)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Fast screening: max 80 epochs, aggressive early stopping at 10 patience
        for epoch in range(80):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_tensor)
                val_loss = criterion(y_val_pred, y_val_tensor).item()
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10:  # Very aggressive for speed
                    model.load_state_dict(best_model_state)
                    break
        
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val_tensor).cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()
        
        ss_res = np.sum((y_val_np - y_pred_val) ** 2)
        ss_tot = np.sum((y_val_np - np.mean(y_val_np)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return r2_score

    # ============ FAST SCREENING: 2-FOLD CV (not 5-fold) ============
    best_overall_r2 = 0
    best_scaler_name = None
    best_config = None
    results_summary = []
    
    for scaler_idx, (scaler_name, scaler_obj) in enumerate(scaler_configs, 1):
        print(f"\n[{scaler_idx}/4] Testing {scaler_name}...", flush=True)
        import sys
        sys.stdout.flush()
        
        if scaler_obj is not None:
            print(f"     Applying scaler...", flush=True)
            x2_scaled = scaler_obj.fit_transform(x2_no_const)
        else:
            x2_scaled = x2_no_const.values
        
        # Use 5-fold CV
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        config_results = []
        
        for config_idx, (hidden_layers, lr, dropouts, batch_size, patience) in enumerate(hyperparameter_configs, 1):
            print(f"     Config {config_idx}/3: {hidden_layers}...", end="", flush=True)
            sys.stdout.flush()
            nn_cv_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(x2_scaled)):
                X_train_fold, X_val_fold = x2_scaled[train_idx], x2_scaled[val_idx]
                y_train_fold, y_val_fold = y2.values[train_idx], y2.values[val_idx]
                
                r2_score = train_nn_model_fast(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                               hidden_layers, lr, dropouts, batch_size, patience, device)
                nn_cv_scores.append(r2_score)
            
            mean_r2 = np.mean(nn_cv_scores)
            std_r2 = np.std(nn_cv_scores)
            
            config_results.append({
                'config_idx': config_idx,
                'hidden_layers': hidden_layers,
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'scores': nn_cv_scores
            })
            
            print(f" R²={mean_r2:.4f}", flush=True)
            sys.stdout.flush()
            
            if mean_r2 > best_overall_r2:
                best_overall_r2 = mean_r2
                best_scaler_name = scaler_name
                best_config = config_results[-1]
        
        config_results.sort(key=lambda x: x['mean_r2'], reverse=True)
        best_for_scaler = config_results[0]['mean_r2']
        results_summary.append({
            'scaler': scaler_name,
            'best_mean_r2': best_for_scaler,
            'best_config': config_results[0]
        })
        
        print(f"     ✓ {scaler_name} Complete - Best R²: {best_for_scaler:.4f}", flush=True)
        sys.stdout.flush()
    
    # ============ FINAL RESULTS ============
    print(f"\n{'='*70}")
    print('==============|SCREENING RESULTS|==============')
    print(f"{'='*70}\n")
    
    print("Scaler Comparison (Ranked by Performance):")
    results_summary.sort(key=lambda x: x['best_mean_r2'], reverse=True)
    for rank, result in enumerate(results_summary, 1):
        config = result['best_config']
        print(f"  {rank}. {result['scaler']:15s} R²={result['best_mean_r2']:.4f}  "
              f"(Config: {config['hidden_layers']}, LR=0.001)")
    
    print(f"\n🏆 WINNER: {best_scaler_name}")
    print(f"   Best Architecture: {best_config['hidden_layers']}")
    print(f"   Mean CV R² (2-fold screening): {best_config['mean_r2']:.4f}")
    print(f"   Fold Scores: {[f'{s:.4f}' for s in best_config['scores']]}")
    
    # Convert to numpy for final summary
    nn_cv_scores = np.array(best_config['scores'])
    print(f"\n[NOTE: This is QUICK SCREENING (2-fold CV, 80 epochs, 3 configs)]")
    print(f"[Recommended: Use {best_scaler_name} for final full model training]\n")
    
    print('==============|Neural Network Screening Complete|==============\n')
else:
    print('SKIPPED: Neural Network Regression (ENABLE_NEURAL_NETWORK = False)\n')
    nn_cv_scores = None


# ============================================================
# XGBoost with Bayesian Optimization (Optuna)
# ============================================================
if ENABLE_XGBOOST_OPTUNA:
    print('\n' + '='*70)
    print('==============|XGBoost Bayesian Optimization (Optuna)|==============\n')
    print(f"Total time budget: ~3 hours for 200+ trials with 5-fold CV...\n")
    
    def objective_xgb(trial):
        """Optuna objective function for XGBoost hyperparameter tuning"""
        params = {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'tree_method': 'hist',  # GPU acceleration
            'device': 'cuda',  # GPU device (replaces deprecated gpu_id)
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        xgb_model = xgb.XGBRegressor(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(xgb_model, x2_no_const, y2, cv=kfold, scoring='r2', n_jobs=-1)
        
        return cv_scores.mean()
    
    # Create Optuna study with TPE sampler (Bayesian optimization)
    sampler = TPESampler(seed=42, n_startup_trials=20)
    study_xgb = optuna.create_study(sampler=sampler, direction='maximize')
    
    print("Starting Bayesian Optimization Search (200 trials, 5-fold CV per trial)...")
    print("This will take approximately 2-3 hours.\n")
    study_xgb.optimize(objective_xgb, n_trials=200, show_progress_bar=True, n_jobs=1)
    
    # Get best trial results
    best_trial_xgb = study_xgb.best_trial
    print(f"\n✓ XGBoost Optuna Search Complete!")
    print(f"\nBest XGBoost Hyperparameters (Trial #{best_trial_xgb.number}):")
    for param, value in best_trial_xgb.params.items():
        if isinstance(value, float):
            print(f"  {param:20s}: {value:.6f}")
        else:
            print(f"  {param:20s}: {value}")
    
    # Final evaluation with 5-fold CV
    xgb_best = xgb.XGBRegressor(
        objective='reg:squarederror',
        booster='gbtree',
        tree_method='hist',  # GPU acceleration
        device='cuda',  # GPU device (replaces deprecated gpu_id)
        learning_rate=best_trial_xgb.params['learning_rate'],
        max_depth=best_trial_xgb.params['max_depth'],
        min_child_weight=best_trial_xgb.params['min_child_weight'],
        subsample=best_trial_xgb.params['subsample'],
        colsample_bytree=best_trial_xgb.params['colsample_bytree'],
        colsample_bylevel=best_trial_xgb.params['colsample_bylevel'],
        reg_alpha=best_trial_xgb.params['reg_alpha'],
        reg_lambda=best_trial_xgb.params['reg_lambda'],
        gamma=best_trial_xgb.params['gamma'],
        n_estimators=best_trial_xgb.params['n_estimators'],
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    xgb_cv_scores = cross_val_score(xgb_best, x2_no_const, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1)
    print(f"\nXGBoost Final CV R² Scores (5 folds): {xgb_cv_scores}")
    print(f"XGBoost Mean CV R²: {xgb_cv_scores.mean():.6f}")
    print(f"XGBoost Std Dev CV R²: {xgb_cv_scores.std():.6f}")
    
    # Optuna trials history
    trials_df = study_xgb.trials_dataframe()
    top_xgb_trials = trials_df.nlargest(5, 'value')
    print(f"\nTop 5 XGBoost trials:")
    for idx, (_, row) in enumerate(top_xgb_trials.iterrows(), 1):
        print(f"  {idx}. Trial #{int(row['number'])}: R² = {row['value']:.6f}")
    
    print('\n==============|XGBoost Optuna Complete|==============\n')
else:
    print('SKIPPED: XGBoost Bayesian Optimization (ENABLE_XGBOOST_OPTUNA = False)\n')
    xgb_cv_scores = None
    xgb_best = None
    best_trial_xgb = None


# ============================================================
# LightGBM with Bayesian Optimization (Optuna)
# ============================================================
if ENABLE_LIGHTGBM_OPTUNA:
    print('\n' + '='*70)
    print('==============|LightGBM Bayesian Optimization (Optuna)|==============\n')
    print(f"Total time budget: ~3 hours for 200 trials with 5-fold CV...\n")
    
    def objective_lgb(trial):
        """Optuna objective function for LightGBM hyperparameter tuning"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'device': 'gpu',  # GPU acceleration
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),  # Wider range
            'max_depth': trial.suggest_int('max_depth', 2, 15),  # Deeper exploration
            'num_leaves': trial.suggest_int('num_leaves', 15, 250),  # More leaves
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 3, 100),  # Wider range
            'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),  # More aggressive
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),  # More aggressive
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 15),  # More frequent bagging
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 100.0, log=True),  # Strong L1 regularization
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 100.0, log=True),  # Strong L2 regularization
            'n_estimators': trial.suggest_int('n_estimators', 50, 800),  # More boosting rounds
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),  # New: split gain threshold
            'verbose': -1,
            'n_jobs': -1
        }
        
        lgb_model = lgb.LGBMRegressor(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(lgb_model, x2_no_const, y2, cv=kfold, scoring='r2', n_jobs=-1)
        
        return cv_scores.mean()
    
    # Create Optuna study with TPE sampler (more aggressive sampling)
    sampler_lgb = TPESampler(seed=42, n_startup_trials=20, constant_liar=True)
    study_lgb = optuna.create_study(sampler=sampler_lgb, direction='maximize')
    
    print("Starting Bayesian Optimization Search (200 trials, 5-fold CV per trial)...")
    print("This will take approximately 2-3 hours.\n")
    study_lgb.optimize(objective_lgb, n_trials=200, show_progress_bar=True, n_jobs=1)
    
    # Get best trial results
    best_trial_lgb = study_lgb.best_trial
    print(f"\n✓ LightGBM Optuna Search Complete!")
    print(f"\nBest LightGBM Hyperparameters (Trial #{best_trial_lgb.number}):")
    for param, value in best_trial_lgb.params.items():
        if isinstance(value, float):
            print(f"  {param:20s}: {value:.6f}")
        else:
            print(f"  {param:20s}: {value}")
    
    # Final evaluation with 5-fold CV
    lgb_best = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        device='gpu',  # GPU acceleration
        learning_rate=best_trial_lgb.params['learning_rate'],
        max_depth=best_trial_lgb.params['max_depth'],
        num_leaves=best_trial_lgb.params['num_leaves'],
        min_data_in_leaf=best_trial_lgb.params['min_data_in_leaf'],
        feature_fraction=best_trial_lgb.params['feature_fraction'],
        bagging_fraction=best_trial_lgb.params['bagging_fraction'],
        bagging_freq=best_trial_lgb.params['bagging_freq'],
        lambda_l1=best_trial_lgb.params['lambda_l1'],
        lambda_l2=best_trial_lgb.params['lambda_l2'],
        n_estimators=best_trial_lgb.params['n_estimators'],
        verbose=-1,
        n_jobs=-1
    )
    
    lgb_cv_scores = cross_val_score(lgb_best, x2_no_const, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1)
    print(f"\nLightGBM Final CV R² Scores (5 folds): {lgb_cv_scores}")
    print(f"LightGBM Mean CV R²: {lgb_cv_scores.mean():.6f}")
    print(f"LightGBM Std Dev CV R²: {lgb_cv_scores.std():.6f}")
    
    # Optuna trials history
    trials_df_lgb = study_lgb.trials_dataframe()
    top_lgb_trials = trials_df_lgb.nlargest(5, 'value')
    print(f"\nTop 5 LightGBM trials:")
    for idx, (_, row) in enumerate(top_lgb_trials.iterrows(), 1):
        print(f"  {idx}. Trial #{int(row['number'])}: R² = {row['value']:.6f}")
    
    print('\n==============|LightGBM Optuna Complete|==============\n')
else:
    print('SKIPPED: LightGBM Bayesian Optimization (ENABLE_LIGHTGBM_OPTUNA = False)\n')
    lgb_cv_scores = None
    lgb_best = None
    best_trial_lgb = None


# ============================================================
# RandomForest with Bayesian Optimization (Optuna)
# ============================================================
if ENABLE_RANDOMFOREST_OPTUNA:
    print('\n' + '='*70)
    print('==============|RandomForest Bayesian Optimization|==============\n')
    print(f"Total time budget: ~2 hours for 100 trials with 5-fold CV...\n")
    
    from sklearn.ensemble import RandomForestRegressor
    
    def objective_rf(trial):
        """Optuna objective function for RandomForest hyperparameter tuning"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 15),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),  # Bootstrap sample size
            'random_state': 42,
            'n_jobs': -1
        }
        
        rf_model = RandomForestRegressor(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf_model, x2_no_const, y2, cv=kfold, scoring='r2', n_jobs=-1)
        
        return cv_scores.mean()
    
    # Create Optuna study with TPE sampler
    sampler_rf = TPESampler(seed=42, n_startup_trials=12)
    study_rf = optuna.create_study(sampler=sampler_rf, direction='maximize')
    
    print("Starting Bayesian Optimization Search (100 trials, 5-fold CV per trial)...")
    print("This will take approximately 1.5-2 hours.\n")
    study_rf.optimize(objective_rf, n_trials=100, show_progress_bar=True, n_jobs=1)
    
    # Get best trial results
    best_trial_rf = study_rf.best_trial
    print(f"\n✓ RandomForest Optuna Search Complete!")
    print(f"\nBest RandomForest Hyperparameters (Trial #{best_trial_rf.number}):")
    for param, value in best_trial_rf.params.items():
        if isinstance(value, float):
            print(f"  {param:20s}: {value:.6f}")
        else:
            print(f"  {param:20s}: {value}")
    
    # Final evaluation with 5-fold CV
    rf_optuna_best = RandomForestRegressor(
        n_estimators=best_trial_rf.params['n_estimators'],
        max_depth=best_trial_rf.params['max_depth'],
        min_samples_split=best_trial_rf.params['min_samples_split'],
        min_samples_leaf=best_trial_rf.params['min_samples_leaf'],
        max_features=best_trial_rf.params['max_features'],
        max_samples=best_trial_rf.params['max_samples'],
        random_state=42,
        n_jobs=-1
    )
    
    rf_optuna_cv_scores = cross_val_score(rf_optuna_best, x2_no_const, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1)
    print(f"\nRandomForest Final CV R² Scores (5 folds): {rf_optuna_cv_scores}")
    print(f"RandomForest Mean CV R²: {rf_optuna_cv_scores.mean():.6f}")
    print(f"RandomForest Std Dev CV R²: {rf_optuna_cv_scores.std():.6f}")
    
    # Optuna trials history
    trials_df_rf = study_rf.trials_dataframe()
    top_rf_trials = trials_df_rf.nlargest(5, 'value')
    print(f"\nTop 5 RandomForest trials:")
    for idx, (_, row) in enumerate(top_rf_trials.iterrows(), 1):
        print(f"  {idx}. Trial #{int(row['number'])}: R² = {row['value']:.6f}")
    
    print('\n==============|RandomForest Optuna Complete|==============\n')
else:
    print('SKIPPED: RandomForest Bayesian Optimization (ENABLE_RANDOMFOREST_OPTUNA = False)\n')
    rf_optuna_cv_scores = None
    rf_optuna_best = None
    best_trial_rf = None


# ============================================================
# HistGradientBoosting with Bayesian Optimization (Optuna)
# ============================================================
if ENABLE_HISTGB_OPTUNA:
    print('\n' + '='*70)
    print('==============|HistGradientBoosting Bayesian Optimization|==============\n')
    print(f"Total time budget: ~2 hours for 100 trials with 5-fold CV...\n")
    
    from sklearn.ensemble import HistGradientBoostingRegressor
    
    def objective_histgb(trial):
        """Optuna objective function for HistGradientBoosting hyperparameter tuning"""
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-5, 10.0, log=True),
            'max_bins': trial.suggest_int('max_bins', 64, 512),
            'n_iter_no_change': None,
            'random_state': 42,
            'n_jobs': -1
        }
        
        hgb_model = HistGradientBoostingRegressor(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(hgb_model, x2_no_const, y2, cv=kfold, scoring='r2', n_jobs=-1)
        
        return cv_scores.mean()
    
    # Create Optuna study with TPE sampler
    sampler_hgb = TPESampler(seed=42, n_startup_trials=12)
    study_hgb = optuna.create_study(sampler=sampler_hgb, direction='maximize')
    
    print("Starting Bayesian Optimization Search (100 trials, 5-fold CV per trial)...")
    print("This will take approximately 1.5-2 hours.\n")
    study_hgb.optimize(objective_histgb, n_trials=100, show_progress_bar=True, n_jobs=1)
    
    # Get best trial results
    best_trial_hgb = study_hgb.best_trial
    print(f"\n✓ HistGradientBoosting Optuna Search Complete!")
    print(f"\nBest HistGradientBoosting Hyperparameters (Trial #{best_trial_hgb.number}):")
    for param, value in best_trial_hgb.params.items():
        if isinstance(value, float):
            print(f"  {param:20s}: {value:.6f}")
        else:
            print(f"  {param:20s}: {value}")
    
    # Final evaluation with 5-fold CV
    hgb_best = HistGradientBoostingRegressor(
        learning_rate=best_trial_hgb.params['learning_rate'],
        max_depth=best_trial_hgb.params['max_depth'],
        max_leaf_nodes=best_trial_hgb.params['max_leaf_nodes'],
        min_samples_leaf=best_trial_hgb.params['min_samples_leaf'],
        l2_regularization=best_trial_hgb.params['l2_regularization'],
        max_bins=best_trial_hgb.params['max_bins'],
        random_state=42,
        n_jobs=-1
    )
    
    hgb_cv_scores = cross_val_score(hgb_best, x2_no_const, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1)
    print(f"\nHistGradientBoosting Final CV R² Scores (5 folds): {hgb_cv_scores}")
    print(f"HistGradientBoosting Mean CV R²: {hgb_cv_scores.mean():.6f}")
    print(f"HistGradientBoosting Std Dev CV R²: {hgb_cv_scores.std():.6f}")
    
    # Optuna trials history
    trials_df_hgb = study_hgb.trials_dataframe()
    top_hgb_trials = trials_df_hgb.nlargest(5, 'value')
    print(f"\nTop 5 HistGradientBoosting trials:")
    for idx, (_, row) in enumerate(top_hgb_trials.iterrows(), 1):
        print(f"  {idx}. Trial #{int(row['number'])}: R² = {row['value']:.6f}")
    
    print('\n==============|HistGradientBoosting Optuna Complete|==============\n')
else:
    print('SKIPPED: HistGradientBoosting Bayesian Optimization (ENABLE_HISTGB_OPTUNA = False)\n')
    hgb_cv_scores = None
    hgb_best = None
    best_trial_hgb = None


# ============================================================
# ExtraTreesRegressor with Bayesian Optimization (Optuna)
# ============================================================
if ENABLE_EXTRATREES_OPTUNA:
    print('\n' + '='*70)
    print('==============|ExtraTreesRegressor Bayesian Optimization|==============\n')
    print(f"Total time budget: ~1.5 hours for 150 trials with 5-fold CV...\n")
    
    from sklearn.ensemble import ExtraTreesRegressor
    
    def objective_extratrees(trial):
        """Optuna objective function for ExtraTreesRegressor hyperparameter tuning"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 10, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,
            'n_jobs': -1
        }
        
        et_model = ExtraTreesRegressor(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(et_model, x2_no_const, y2, cv=kfold, scoring='r2', n_jobs=-1)
        
        return cv_scores.mean()
    
    # Create Optuna study with TPE sampler
    sampler_et = TPESampler(seed=42, n_startup_trials=15)
    study_et = optuna.create_study(sampler=sampler_et, direction='maximize')
    
    print("Starting Bayesian Optimization Search (150 trials, 5-fold CV per trial)...")
    print("This will take approximately 1.5-2 hours.\n")
    study_et.optimize(objective_extratrees, n_trials=150, show_progress_bar=True, n_jobs=1)
    
    # Get best trial results
    best_trial_et = study_et.best_trial
    print(f"\n✓ ExtraTreesRegressor Optuna Search Complete!")
    print(f"\nBest ExtraTreesRegressor Hyperparameters (Trial #{best_trial_et.number}):")
    for param, value in best_trial_et.params.items():
        if isinstance(value, float):
            print(f"  {param:20s}: {value:.6f}")
        else:
            print(f"  {param:20s}: {value}")
    
    # Final evaluation with 5-fold CV
    et_best = ExtraTreesRegressor(
        n_estimators=best_trial_et.params['n_estimators'],
        max_depth=best_trial_et.params['max_depth'],
        min_samples_split=best_trial_et.params['min_samples_split'],
        min_samples_leaf=best_trial_et.params['min_samples_leaf'],
        max_features=best_trial_et.params['max_features'],
        bootstrap=best_trial_et.params['bootstrap'],
        random_state=42,
        n_jobs=-1
    )
    
    et_cv_scores = cross_val_score(et_best, x2_no_const, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1)
    print(f"\nExtraTreesRegressor Final CV R² Scores (5 folds): {et_cv_scores}")
    print(f"ExtraTreesRegressor Mean CV R²: {et_cv_scores.mean():.6f}")
    print(f"ExtraTreesRegressor Std Dev CV R²: {et_cv_scores.std():.6f}")
    
    # Optuna trials history
    trials_df_et = study_et.trials_dataframe()
    top_et_trials = trials_df_et.nlargest(5, 'value')
    print(f"\nTop 5 ExtraTreesRegressor trials:")
    for idx, (_, row) in enumerate(top_et_trials.iterrows(), 1):
        print(f"  {idx}. Trial #{int(row['number'])}: R² = {row['value']:.6f}")
    
    print('\n==============|ExtraTreesRegressor Optuna Complete|==============\n')
else:
    print('SKIPPED: ExtraTreesRegressor Bayesian Optimization (ENABLE_EXTRATREES_OPTUNA = False)\n')
    et_cv_scores = None
    et_best = None
    best_trial_et = None


# ============================================================
# CatBoost with Bayesian Optimization (Optuna)
# ============================================================
if ENABLE_CATBOOST_OPTUNA:
    print('\n' + '='*70)
    print('==============|CatBoost Bayesian Optimization (Optuna)|==============\n')
    print(f"Total time budget: ~1.5 hours for 120 trials with 5-fold CV...\n")
    
    import catboost as cb
    
    def objective_catboost(trial):
        """Optuna objective function for CatBoost hyperparameter tuning"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'task_type': 'GPU',  # GPU acceleration
            'verbose': False,
            'thread_count': -1
        }
        
        cb_model = cb.CatBoostRegressor(**params)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(cb_model, x2_no_const, y2, cv=kfold, scoring='r2', n_jobs=-1)
        
        return cv_scores.mean()
    
    # Create Optuna study with TPE sampler
    sampler_cb = TPESampler(seed=42, n_startup_trials=12)
    study_cb = optuna.create_study(sampler=sampler_cb, direction='maximize')
    
    print("Starting Bayesian Optimization Search (120 trials, 5-fold CV per trial)...")
    print("This will take approximately 1.5-2 hours.\n")
    study_cb.optimize(objective_catboost, n_trials=120, show_progress_bar=True, n_jobs=1)
    
    # Get best trial results
    best_trial_cb = study_cb.best_trial
    print(f"\n✓ CatBoost Optuna Search Complete!")
    print(f"\nBest CatBoost Hyperparameters (Trial #{best_trial_cb.number}):")
    for param, value in best_trial_cb.params.items():
        if isinstance(value, float):
            print(f"  {param:20s}: {value:.6f}")
        else:
            print(f"  {param:20s}: {value}")
    
    # Final evaluation with 5-fold CV
    cb_best = cb.CatBoostRegressor(
        iterations=best_trial_cb.params['iterations'],
        learning_rate=best_trial_cb.params['learning_rate'],
        depth=best_trial_cb.params['depth'],
        l2_leaf_reg=best_trial_cb.params['l2_leaf_reg'],
        subsample=best_trial_cb.params['subsample'],
        colsample_bylevel=best_trial_cb.params['colsample_bylevel'],
        random_strength=best_trial_cb.params['random_strength'],
        task_type='GPU',  # GPU acceleration
        verbose=False,
        thread_count=-1
    )
    
    cb_cv_scores = cross_val_score(cb_best, x2_no_const, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1)
    print(f"\nCatBoost Final CV R² Scores (5 folds): {cb_cv_scores}")
    print(f"CatBoost Mean CV R²: {cb_cv_scores.mean():.6f}")
    print(f"CatBoost Std Dev CV R²: {cb_cv_scores.std():.6f}")
    
    # Optuna trials history
    trials_df_cb = study_cb.trials_dataframe()
    top_cb_trials = trials_df_cb.nlargest(5, 'value')
    print(f"\nTop 5 CatBoost trials:")
    for idx, (_, row) in enumerate(top_cb_trials.iterrows(), 1):
        print(f"  {idx}. Trial #{int(row['number'])}: R² = {row['value']:.6f}")
    
    print('\n==============|CatBoost Optuna Complete|==============\n')
else:
    print('SKIPPED: CatBoost Bayesian Optimization (ENABLE_CATBOOST_OPTUNA = False)\n')
    cb_cv_scores = None
    cb_best = None
    best_trial_cb = None


# ============================================================
# Weighted Ensemble (Voting Regressor with optimized weights)
# ============================================================
if ENABLE_WEIGHTED_ENSEMBLE and xgb_best is not None and cb_best is not None and lgb_best is not None:
    print('\n' + '='*70)
    print('==============|Weighted Ensemble Voting|==============\n')
    
    from sklearn.ensemble import VotingRegressor
    
    # Create weighted voting ensemble with best models (LightGBM, XGBoost, CatBoost as base)
    ensemble_models = [
        ('xgb', xgb_best),
        ('cb', cb_best),
        ('lgb', lgb_best),
    ]
    
    ensemble_models = [m for m in ensemble_models if m is not None]
    
    print(f"Base models in ensemble: {[name for name, _ in ensemble_models]}\n")
    
    # Test different weight combinations
    print("Testing weight combinations to optimize ensemble...\n")
    
    best_ensemble_score = 0
    best_weights = None
    
    n_models = len(ensemble_models)
    
    # Generate diverse weight combinations for 3-model ensemble
    test_weights_list = [
        [1.0/3, 1.0/3, 1.0/3],  # Equal weights
        [0.4, 0.3, 0.3],  # LightGBM dominant
        [0.3, 0.4, 0.3],  # XGBoost dominant
        [0.3, 0.3, 0.4],  # CatBoost dominant
        [0.35, 0.35, 0.3],  # LGB+XGB dominant
        [0.35, 0.3, 0.35],  # LGB+CB dominant
        [0.33, 0.34, 0.33],  # Near equal
    ]
    
    weight_results = []
    
    kfold_ensemble = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for weights in test_weights_list:
        print(f"  Testing weights {[f'{w:.2f}' for w in weights]}...", end='', flush=True)
        
        voter = VotingRegressor(
            estimators=ensemble_models,
            weights=weights
        )
        
        scores = cross_val_score(voter, x2_no_const, y2, cv=kfold_ensemble, scoring='r2', n_jobs=-1)
        mean_score = scores.mean()
        weight_results.append((weights, mean_score, scores.std()))
        
        print(f" R² = {mean_score:.6f}")
        
        if mean_score > best_ensemble_score:
            best_ensemble_score = mean_score
            best_weights = weights
    
    # Create final ensemble with best weights
    final_ensemble = VotingRegressor(
        estimators=ensemble_models,
        weights=best_weights
    )
    
    ensemble_cv_scores = cross_val_score(final_ensemble, x2_no_const, y2, cv=kfold_ensemble, scoring='r2', n_jobs=-1)
    
    print(f"\n✓ Weighted Ensemble Complete!")
    print(f"\nBest Weights: {best_weights}")
    for i, (name, _) in enumerate(ensemble_models):
        print(f"  {name.upper():15s}: {best_weights[i]:.4f}")
    print(f"\nEnsemble CV R² Scores (5 folds): {ensemble_cv_scores}")
    print(f"Ensemble Mean CV R²: {ensemble_cv_scores.mean():.6f}")
    print(f"Ensemble Std Dev CV R²: {ensemble_cv_scores.std():.6f}")
    
    print(f"\nAll weight combinations tested:")
    weight_results.sort(key=lambda x: x[1], reverse=True)
    for rank, (weights, score, std) in enumerate(weight_results, 1):
        print(f"  {rank}. {[f'{w:.2f}' for w in weights]} → R² = {score:.6f} ± {std:.6f}")
    
    print('\n==============|Weighted Ensemble Complete|==============\n')
else:
    print('SKIPPED: Weighted Ensemble (requirements not met)\n')
    ensemble_cv_scores = None


# ============================================================
# Stacking Ensemble (Meta-Learner)
# ============================================================
if ENABLE_STACKING and ENABLE_RANDOM_FOREST and ENABLE_XGBOOST_OPTUNA and ENABLE_LIGHTGBM_OPTUNA:
    print('\n' + '='*70)
    print('==============|Stacking Ensemble with Meta-Learner|==============\n')
    
    # Use best models as base learners
    base_models = []
    base_model_names = []
    
    if rf_best is not None:
        base_models.append(('rf', rf_best))
        base_model_names.append('Random Forest')
    if xgb_best is not None:
        base_models.append(('xgb', xgb_best))
        base_model_names.append('XGBoost')
    if lgb_best is not None:
        base_models.append(('lgb', lgb_best))
        base_model_names.append('LightGBM')
    if gb_best is not None:
        base_models.append(('gb', gb_best))
        base_model_names.append('Gradient Boosting')
    
    if len(base_models) >= 2:
        print(f"Base models for stacking: {', '.join(base_model_names)}\n")
        print("Generating meta-features with 5-fold cross-validation...\n")
        
        kfold_stack = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features_train = np.zeros((len(x2_no_const), len(base_models)))
        meta_features_test = np.zeros((len(x2_no_const), len(base_models)))
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold_stack.split(x2_no_const)):
            print(f"  Fold {fold_idx + 1}/5...", end='', flush=True)
            X_train_fold, X_test_fold = x2_no_const.iloc[train_idx], x2_no_const.iloc[test_idx]
            y_train_fold = y2.iloc[train_idx]
            
            for model_idx, (model_name, model) in enumerate(base_models):
                # Clone and train base model on fold
                if model_name == 'rf':
                    clone_model = RandomForestRegressor(
                        n_estimators=model.n_estimators,
                        max_depth=model.max_depth,
                        min_samples_split=model.min_samples_split,
                        min_samples_leaf=model.min_samples_leaf,
                        max_features=model.max_features,
                        random_state=42,
                        n_jobs=-1
                    )
                elif model_name == 'xgb':
                    clone_model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        learning_rate=model.learning_rate,
                        max_depth=model.max_depth,
                        n_estimators=model.n_estimators,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0
                    )
                elif model_name == 'lgb':
                    clone_model = lgb.LGBMRegressor(
                        learning_rate=model.learning_rate,
                        max_depth=model.max_depth,
                        n_estimators=model.n_estimators,
                        verbose=-1,
                        n_jobs=-1
                    )
                elif model_name == 'gb':
                    clone_model = GradientBoostingRegressor(
                        n_estimators=model.n_estimators,
                        learning_rate=model.learning_rate,
                        max_depth=model.max_depth,
                        random_state=42
                    )
                
                clone_model.fit(X_train_fold, y_train_fold)
                meta_features_train[test_idx, model_idx] = clone_model.predict(X_test_fold)
            
            print(" ✓")
        
        # Train all base models on full data to generate test meta-features
        print(f"  Training all base models on full data...")
        for model_idx, (model_name, model) in enumerate(base_models):
            model.fit(x2_no_const, y2)
            meta_features_test[:, model_idx] = model.predict(x2_no_const)
        
        # Train meta-learner (Ridge regression) on meta-features
        print(f"\nTraining meta-learner (Ridge Regression) on meta-features...\n")
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(meta_features_train, y2)
        
        # Evaluate stacking model with k-fold
        stack_predictions = np.zeros_like(y2, dtype=float)
        kfold_eval = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, test_idx in kfold_eval.split(x2_no_const):
            X_train_fold, X_test_fold = x2_no_const.iloc[train_idx], x2_no_const.iloc[test_idx]
            y_train_fold = y2.iloc[train_idx]
            
            meta_train = meta_features_train[train_idx]
            meta_test = meta_features_train[test_idx]
            
            meta_learner_fold = Ridge(alpha=1.0)
            meta_learner_fold.fit(meta_train, y_train_fold)
            stack_predictions[test_idx] = meta_learner_fold.predict(meta_test)
        
        # Calculate stacking CV scores
        stack_r2_scores = []
        for train_idx, test_idx in kfold_eval.split(x2_no_const):
            meta_train = meta_features_train[train_idx]
            meta_test = meta_features_train[test_idx]
            y_test_fold = y2.iloc[test_idx]
            
            meta_learner_fold = Ridge(alpha=1.0)
            meta_learner_fold.fit(meta_train, y2.iloc[train_idx])
            fold_r2 = meta_learner_fold.score(meta_test, y_test_fold)
            stack_r2_scores.append(fold_r2)
        
        stack_cv_scores = np.array(stack_r2_scores)
        
        print(f"Stacking Ensemble CV R² Scores (5 folds): {stack_cv_scores}")
        print(f"Stacking Mean CV R²: {stack_cv_scores.mean():.6f}")
        print(f"Stacking Std Dev CV R²: {stack_cv_scores.std():.6f}")
        
        print(f"\nMeta-Learner Coefficients (base model weights):")
        for model_name, coef in zip(base_model_names, meta_learner.coef_):
            print(f"  {model_name:20s}: {coef:.6f}")
        print(f"  Intercept             : {meta_learner.intercept_:.6f}")
        
        print('\n==============|Stacking Ensemble Complete|==============\n')
    else:
        print(f"❌ Not enough base models for stacking (need ≥2, have {len(base_models)})\n")
        stack_cv_scores = None
else:
    print('SKIPPED: Stacking Ensemble (requirements not met)\n')
    stack_cv_scores = None


# ============================================================
# Feature Importance Analysis - Identify Weak Features
# ============================================================
print('\n' + '='*70)
print('==============|FEATURE IMPORTANCE ANALYSIS|==============')
print('='*70 + '\n')

feature_importance_dict = {}

# Extract feature importance from LightGBM
if ENABLE_LIGHTGBM_OPTUNA and lgb_best is not None:
    print("Analyzing LightGBM Feature Importance...\n")
    lgb_best.fit(x2_no_const, y2)
    lgb_importance = pd.DataFrame({
        'feature': x2_no_const.columns,
        'importance': lgb_best.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 20 LightGBM Features:")
    print(lgb_importance.head(20).to_string(index=False))
    
    # Store for averaging
    for feat, imp in zip(lgb_importance['feature'], lgb_importance['importance']):
        if feat not in feature_importance_dict:
            feature_importance_dict[feat] = []
        feature_importance_dict[feat].append(imp)
    
    print(f"\nBottom 10 LightGBM Features (candidates for removal):")
    print(lgb_importance.tail(10).to_string(index=False))
    print()

# Extract feature importance from XGBoost
if ENABLE_XGBOOST_OPTUNA and xgb_best is not None:
    print("Analyzing XGBoost Feature Importance...\n")
    xgb_best.fit(x2_no_const, y2)
    xgb_importance = pd.DataFrame({
        'feature': x2_no_const.columns,
        'importance': xgb_best.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 20 XGBoost Features:")
    print(xgb_importance.head(20).to_string(index=False))
    
    # Store for averaging
    for feat, imp in zip(xgb_importance['feature'], xgb_importance['importance']):
        if feat not in feature_importance_dict:
            feature_importance_dict[feat] = []
        feature_importance_dict[feat].append(imp)
    
    print(f"\nBottom 10 XGBoost Features (candidates for removal):")
    print(xgb_importance.tail(10).to_string(index=False))
    print()

# Extract feature importance from CatBoost
if ENABLE_CATBOOST_OPTUNA and cb_best is not None:
    print("Analyzing CatBoost Feature Importance...\n")
    cb_best.fit(x2_no_const, y2, verbose=False)
    cb_importance = pd.DataFrame({
        'feature': x2_no_const.columns,
        'importance': cb_best.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 20 CatBoost Features:")
    print(cb_importance.head(20).to_string(index=False))
    
    # Store for averaging
    for feat, imp in zip(cb_importance['feature'], cb_importance['importance']):
        if feat not in feature_importance_dict:
            feature_importance_dict[feat] = []
        feature_importance_dict[feat].append(imp)
    
    print(f"\nBottom 10 CatBoost Features (candidates for removal):")
    print(cb_importance.tail(10).to_string(index=False))
    print()

# Calculate average feature importance across models
if feature_importance_dict:
    avg_importance = {feat: np.mean(imps) for feat, imps in feature_importance_dict.items()}
    avg_importance_df = pd.DataFrame(list(avg_importance.items()), 
                                     columns=['feature', 'avg_importance']).sort_values('avg_importance', ascending=False)
    
    print('\n' + '='*70)
    print("AVERAGE FEATURE IMPORTANCE (across all models):")
    print('='*70 + '\n')
    print("Top 25 Most Important Features:")
    print(avg_importance_df.head(25).to_string(index=False))
    
    print(f"\n\nBottom 15 Least Important Features (candidates for removal):")
    removal_candidates = avg_importance_df.tail(15)
    print(removal_candidates.to_string(index=False))
    
    print(f"\n⚠️  FEATURE REMOVAL SUGGESTIONS:")
    print(f"Features with importance < {avg_importance_df['avg_importance'].quantile(0.1):.6f} might be removed:")
    weak_features = avg_importance_df[avg_importance_df['avg_importance'] < avg_importance_df['avg_importance'].quantile(0.1)]
    for idx, row in weak_features.iterrows():
        print(f"  - {row['feature']:30s} (avg importance: {row['avg_importance']:.8f})")
    
    print(f"\n📊 Feature Importance Statistics:")
    print(f"  Total features: {len(avg_importance_df)}")
    print(f"  Mean importance: {avg_importance_df['avg_importance'].mean():.8f}")
    print(f"  Std dev: {avg_importance_df['avg_importance'].std():.8f}")
    print(f"  Min importance: {avg_importance_df['avg_importance'].min():.8f}")
    print(f"  Max importance: {avg_importance_df['avg_importance'].max():.8f}")

print('\n' + '='*70)
print('==============|END FEATURE IMPORTANCE ANALYSIS|==============')
print('='*70 + '\n')


# Updated Model Comparison Summary
print('\n' + '='*70)
print('==============|FINAL MODEL COMPARISON SUMMARY|==============')
print('='*70 + '\n')

if ENABLE_LINEAR_REGRESSION and linear_cv_scores is not None:
    print(f"Linear Regression (Model 2)     - Mean CV R²: {linear_cv_scores.mean():.6f} ± {linear_cv_scores.std():.6f}")
if ENABLE_RIDGE_REGRESSION and ridge_cv_scores is not None:
    print(f"Ridge Regression (Model 3)      - Mean CV R²: {ridge_cv_scores.mean():.6f} ± {ridge_cv_scores.std():.6f}")
if ENABLE_RANDOM_FOREST and rf_cv_scores is not None:
    print(f"Random Forest (Model 5)         - Mean CV R²: {rf_cv_scores.mean():.6f} ± {rf_cv_scores.std():.6f}")
if ENABLE_GRADIENT_BOOSTING and gb_cv_scores is not None:
    print(f"Gradient Boosting (Model 6)     - Mean CV R²: {gb_cv_scores.mean():.6f} ± {gb_cv_scores.std():.6f}")
if ENABLE_BAGGING_REGRESSOR and bagging_cv_scores is not None:
    print(f"Bagging Regressor (Model 7)     - Mean CV R²: {bagging_cv_scores.mean():.6f} ± {bagging_cv_scores.std():.6f}")
if ENABLE_NEURAL_NETWORK and nn_cv_scores is not None:
    print(f"Neural Network (Model 8)        - Mean CV R²: {nn_cv_scores.mean():.6f} ± {nn_cv_scores.std():.6f}")
if ENABLE_XGBOOST_OPTUNA and xgb_cv_scores is not None:
    print(f"XGBoost (Optuna, Model 9)       - Mean CV R²: {xgb_cv_scores.mean():.6f} ± {xgb_cv_scores.std():.6f}")
if ENABLE_LIGHTGBM_OPTUNA and lgb_cv_scores is not None:
    print(f"LightGBM (Optuna, Model 10)     - Mean CV R²: {lgb_cv_scores.mean():.6f} ± {lgb_cv_scores.std():.6f}")
if ENABLE_EXTRATREES_OPTUNA and et_cv_scores is not None:
    print(f"ExtraTreesRegressor (Model 11)  - Mean CV R²: {et_cv_scores.mean():.6f} ± {et_cv_scores.std():.6f}")
if ENABLE_CATBOOST_OPTUNA and cb_cv_scores is not None:
    print(f"CatBoost (Optuna, Model 12)     - Mean CV R²: {cb_cv_scores.mean():.6f} ± {cb_cv_scores.std():.6f}")
if ENABLE_WEIGHTED_ENSEMBLE and ensemble_cv_scores is not None:
    print(f"Weighted Ensemble (Model 15)    - Mean CV R²: {ensemble_cv_scores.mean():.6f} ± {ensemble_cv_scores.std():.6f}")
if ENABLE_STACKING and stack_cv_scores is not None:
    print(f"Stacking Ensemble (Model 16)    - Mean CV R²: {stack_cv_scores.mean():.6f} ± {stack_cv_scores.std():.6f}")

# Determine best model
models_comparison = []
if ENABLE_LINEAR_REGRESSION and linear_cv_scores is not None:
    models_comparison.append(('Linear', linear_cv_scores.mean()))
if ENABLE_RIDGE_REGRESSION and ridge_cv_scores is not None:
    models_comparison.append(('Ridge', ridge_cv_scores.mean()))
if ENABLE_RANDOM_FOREST and rf_cv_scores is not None:
    models_comparison.append(('Random Forest', rf_cv_scores.mean()))
if ENABLE_GRADIENT_BOOSTING and gb_cv_scores is not None:
    models_comparison.append(('Gradient Boosting', gb_cv_scores.mean()))
if ENABLE_BAGGING_REGRESSOR and bagging_cv_scores is not None:
    models_comparison.append(('Bagging Regressor', bagging_cv_scores.mean()))
if ENABLE_NEURAL_NETWORK and nn_cv_scores is not None:
    models_comparison.append(('Neural Network', nn_cv_scores.mean()))
if ENABLE_XGBOOST_OPTUNA and xgb_cv_scores is not None:
    models_comparison.append(('XGBoost', xgb_cv_scores.mean()))
if ENABLE_LIGHTGBM_OPTUNA and lgb_cv_scores is not None:
    models_comparison.append(('LightGBM', lgb_cv_scores.mean()))
if ENABLE_EXTRATREES_OPTUNA and et_cv_scores is not None:
    models_comparison.append(('ExtraTreesRegressor', et_cv_scores.mean()))
if ENABLE_CATBOOST_OPTUNA and cb_cv_scores is not None:
    models_comparison.append(('CatBoost', cb_cv_scores.mean()))
if ENABLE_WEIGHTED_ENSEMBLE and ensemble_cv_scores is not None:
    models_comparison.append(('Weighted Ensemble', ensemble_cv_scores.mean()))
if ENABLE_STACKING and stack_cv_scores is not None:
    models_comparison.append(('Stacking Ensemble', stack_cv_scores.mean()))

if models_comparison:
    models_comparison.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*70}")
    print("TOP 3 BEST MODELS:")
    print('='*70)
    for rank, (name, r2) in enumerate(models_comparison[:3], 1):
        print(f"  {rank}. {name:30s} R² = {r2:.6f}")
    
    best_model = models_comparison[0]
    print(f"\n🏆 WINNER: {best_model[0]} with R² = {best_model[1]:.6f}")
    print('='*70 + '\n')
else:
    print("\n⚠️  No models enabled for comparison.\n")

# Feature Importance Analysis
if ENABLE_FEATURE_IMPORTANCE and ENABLE_LINEAR_REGRESSION and results2 is not None:
    print('==============|Feature Importance Analysis|==============\n')
    print(f"Top 15 Most Important Features (by absolute coefficient magnitude):\n")

    # Get the OLS model coefficients (excluding constant)
    coef_importance = pd.DataFrame({
        'feature': features2,
        'coefficient': results2.params[1:].values  # Skip constant term
    }).sort_values('coefficient', key=abs, ascending=False)

    for idx, (_, row) in enumerate(coef_importance.head(15).iterrows(), 1):
        print(f"{idx:2d}. {row['feature']:25s} -> Coef: {row['coefficient']:12.6f}")

    print('\n==============|Feature Importance Complete|==============\n')


# Random Forest Feature Importance
if ENABLE_FEATURE_IMPORTANCE and ENABLE_RANDOM_FOREST and rf_final is not None:
    print('==============|Random Forest Feature Importance Analysis|==============\n')
    print(f"Top 15 Most Important Features (by Random Forest importance):\n")

    # Train final RF model on all data to get feature importances
    rf_final_trained = RandomForestRegressor(
        n_estimators=rf_grid.best_params_['n_estimators'],
        max_depth=rf_grid.best_params_['max_depth'],
        min_samples_split=rf_grid.best_params_['min_samples_split'],
        min_samples_leaf=rf_grid.best_params_['min_samples_leaf'],
        max_features=rf_grid.best_params_['max_features'],
        random_state=42,
        n_jobs=12
    )
    rf_final_trained.fit(x2_no_const, y2)

    rf_importance = pd.DataFrame({
        'feature': features2,
        'importance': rf_final_trained.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, (_, row) in enumerate(rf_importance.head(15).iterrows(), 1):
        print(f"{idx:2d}. {row['feature']:25s} -> Importance: {row['importance']:12.6f}")

    print('\n==============|Random Forest Feature Importance Complete|==============\n')

    # Permutation Importance Analysis
    if ENABLE_FEATURE_IMPORTANCE:
        print('==============|Permutation Importance Analysis|==============\n')
        print("Computing permutation importance (measures R² drop when feature is shuffled)...\n")

        perm_importance = permutation_importance(rf_final_trained, x2_no_const, y2, n_repeats=10, random_state=42, n_jobs=12, scoring='r2')

        perm_importance_df = pd.DataFrame({
            'feature': features2,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)

        print(f"Top 15 Most Important Features (by Permutation Importance):\n")
        print("Note: Positive values = R² decrease when feature is shuffled (more important)")
        print("      Negative values = minimal/no impact (feature may be redundant)\n")

        for idx, (_, row) in enumerate(perm_importance_df.head(15).iterrows(), 1):
            print(f"{idx:2d}. {row['feature']:25s} -> Mean: {row['importance_mean']:10.6f} ± {row['importance_std']:10.6f}")

        print('\n' + '='*70)
        print("Features with NEGATIVE or near-zero permutation importance are candidates")
        print("for removal, as they don't meaningfully impact model performance.")
        print('='*70 + '\n')

        # Show features with minimal importance (potential candidates for removal)
        minimal_importance = perm_importance_df[perm_importance_df['importance_mean'] < 0.001]
        if len(minimal_importance) > 0:
            print(f"⚠️  Candidate Features for Removal (importance < 0.001):\n")
            for idx, (_, row) in enumerate(minimal_importance.iterrows(), 1):
                print(f"  {idx}. {row['feature']:25s} -> {row['importance_mean']:10.6f}")
        else:
            print("✓ No features with near-zero permutation importance found.\n")

        print('==============|Permutation Importance Complete|==============\n')

