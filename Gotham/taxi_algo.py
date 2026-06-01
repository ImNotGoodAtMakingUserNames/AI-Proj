# Imports
from enum import IntEnum
from dataclasses import dataclass
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor

from ISLP.models import (ModelSpec as MS, summarize)


# ============== MODEL TOGGLES ==============
# Set to True to run a model, False to skip it
ENABLE_LINEAR_REGRESSION = False
ENABLE_RIDGE_REGRESSION = False
ENABLE_RANDOM_FOREST = False
ENABLE_GRADIENT_BOOSTING = True
ENABLE_FEATURE_IMPORTANCE = True
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

def getEndHour(end_time):
    """Extract hour from end_time"""
    if pd.isna(end_time):
        raise ValueError("end_time is NaN")
    return int(end_time.hour)

def getEndMinute(end_time):
    """Extract minute from end_time"""
    if pd.isna(end_time):
        raise ValueError("end_time is NaN")
    return int(end_time.minute)

def getEndSecond(end_time):
    """Extract second from end_time"""
    if pd.isna(end_time):
        raise ValueError("end_time is NaN")
    return int(end_time.second)

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
features2 = [
    # Original features
    'start_hour', 'start_minute', 'start_day', 'start_month', 
    'x2xDistance', 'dayOfWeek', 'end_hour', 
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


# # Lasso Regression with GridSearchCV
# print('==============|Lasso Regression - Model 4|==============\n')
# print('Note: Lasso may show ConvergenceWarning - this is normal with many features. Increasing iterations to handle engineered features...\n')
# lasso_alphas = np.logspace(-4, 1, 150)  # Expanded range: 0.0001 to 10
# lasso_model = Lasso(max_iter=200000, tol=1e-3, warm_start=False)

# lasso_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('lasso', Lasso(max_iter=200000, tol=1e-3, warm_start=False))
# ])

# lasso_grid = GridSearchCV(lasso_pipeline, {'lasso__alpha': lasso_alphas}, cv=5, scoring='r2', n_jobs=12)
# lasso_grid.fit(x2_no_const, y2)

# print(f"Best Lasso Alpha: {lasso_grid.best_params_['lasso__alpha']:.6f}")
# print(f"Best Lasso CV R² Score: {lasso_grid.best_score_:.6f}")

# # Evaluate best Lasso model with k-fold
# lasso_best = Lasso(alpha=lasso_grid.best_params_['lasso__alpha'], max_iter=200000, tol=1e-3, warm_start=False)
# lasso_cv_scores = cross_val_score(lasso_best, x2_scaled, y2, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
# print(f"Lasso CV R² Scores (5 folds): {lasso_cv_scores}")
# print(f"Lasso Mean CV R²: {lasso_cv_scores.mean():.6f}")
# print(f"Lasso Std Dev CV R²: {lasso_cv_scores.std():.6f}")

# # Show top 10 alpha values tested
# top_alphas_lasso = sorted(zip(lasso_grid.cv_results_['param_lasso__alpha'], lasso_grid.cv_results_['mean_test_score']), key=lambda x: x[1], reverse=True)[:10]
# print(f"\nTop 10 Lasso Alpha values tested:")
# for alpha, score in top_alphas_lasso:
#     print(f"  Alpha: {alpha:.6f} -> R²: {score:.6f}")
# print('==============|Lasso Regression Complete|==============\n')


# Model Comparison Summary
print('==============|Model Comparison Summary|==============')
if ENABLE_LINEAR_REGRESSION and linear_cv_scores is not None:
    print(f"Linear Regression (Model 2)     - Mean CV R²: {linear_cv_scores.mean():.6f} ± {linear_cv_scores.std():.6f}")
if ENABLE_RIDGE_REGRESSION and ridge_cv_scores is not None:
    print(f"Ridge Regression (Model 3)      - Mean CV R²: {ridge_cv_scores.mean():.6f} ± {ridge_cv_scores.std():.6f}")
# print(f"Lasso Regression (Model 4)      - Mean CV R²: {lasso_cv_scores.mean():.6f} ± {lasso_cv_scores.std():.6f}")
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
    rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='r2', n_jobs=12, verbose=1)

    print("Training Random Forest with GridSearchCV (12 combinations, 3-fold CV)...\n")
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
    gb_grid = RandomizedSearchCV(gb_model, gb_param_grid, n_iter=12, cv=2, scoring='r2', n_jobs=12, verbose=1, random_state=42)

    print("Training Gradient Boosting with RandomizedSearchCV (12 random iterations, 2-fold CV)...\n")
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



# Updated Model Comparison Summary
print('==============|Final Model Comparison Summary|==============')
if ENABLE_LINEAR_REGRESSION and linear_cv_scores is not None:
    print(f"Linear Regression (Model 2)     - Mean CV R²: {linear_cv_scores.mean():.6f} ± {linear_cv_scores.std():.6f}")
if ENABLE_RIDGE_REGRESSION and ridge_cv_scores is not None:
    print(f"Ridge Regression (Model 3)      - Mean CV R²: {ridge_cv_scores.mean():.6f} ± {ridge_cv_scores.std():.6f}")
# print(f"Lasso Regression (Model 4)      - Mean CV R²: {lasso_cv_scores.mean():.6f} ± {lasso_cv_scores.std():.6f}")
if ENABLE_RANDOM_FOREST and rf_cv_scores is not None:
    print(f"Random Forest (Model 5)         - Mean CV R²: {rf_cv_scores.mean():.6f} ± {rf_cv_scores.std():.6f}")
if ENABLE_GRADIENT_BOOSTING and gb_cv_scores is not None:
    print(f"Gradient Boosting (Model 6)     - Mean CV R²: {gb_cv_scores.mean():.6f} ± {gb_cv_scores.std():.6f}")

# Determine best model
if ENABLE_LINEAR_REGRESSION or ENABLE_RIDGE_REGRESSION or ENABLE_RANDOM_FOREST or ENABLE_GRADIENT_BOOSTING:
    models_comparison = []
    if ENABLE_LINEAR_REGRESSION and linear_cv_scores is not None:
        models_comparison.append(('Linear', linear_cv_scores.mean()))
    if ENABLE_RIDGE_REGRESSION and ridge_cv_scores is not None:
        models_comparison.append(('Ridge', ridge_cv_scores.mean()))
    # ('Lasso', lasso_cv_scores.mean()),
    if ENABLE_RANDOM_FOREST and rf_cv_scores is not None:
        models_comparison.append(('Random Forest', rf_cv_scores.mean()))
    if ENABLE_GRADIENT_BOOSTING and gb_cv_scores is not None:
        models_comparison.append(('Gradient Boosting', gb_cv_scores.mean()))
    
    if models_comparison:
        best_model = max(models_comparison, key=lambda x: x[1])
        print(f"\n🏆 Best Model: {best_model[0]} with R² = {best_model[1]:.6f}")
print('==============|Comparison Complete|==============\n')

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

