# Imports
from enum import IntEnum
from dataclasses import dataclass
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Neural Network imports
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

sample_frac = 0.3
# ============== MODEL TOGGLES ==============
ENABLE_LINEAR_REGRESSION = False
ENABLE_RIDGE_REGRESSION = False
ENABLE_LASSO_REGRESSION = False
ENABLE_NEURAL_NETWORK = False

    # Tree-based models
ENABLE_RANDOM_FOREST = False
ENABLE_DECISION_TREE = True
ENABLE_GRADIENT_BOOSTING = True
ENABLE_ADABOOST = True
ENABLE_XGBOOST = True
ENABLE_CATBOOST = True
ENABLE_LIGHTGBM = True
# ============================================

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

def setup_features(df):
# Read Data
    taxi = df.sample(frac=sample_frac, random_state=42).copy()

    print('==============|Training Data Imported|==============')
    print(f'Using {sample_frac*100}% of training data ({len(taxi)} rows out of {len(df)} total rows)\n')

    # Starting values
    taxi['start_hour'] = taxi['pickup_datetime'].apply(getStartHour)
    taxi['start_minute'] = taxi['pickup_datetime'].apply(getStartMinute)
    taxi['start_second'] = taxi['pickup_datetime'].apply(getStartSecond)
    taxi['start_day'] = taxi['pickup_datetime'].apply(getStartDay)
    taxi['start_month'] = taxi['pickup_datetime'].apply(getStartMonth)
    taxi['start_year'] = taxi['pickup_datetime'].apply(getStartYear)

    # Time-based values
    taxi['dayOfWeek'] = taxi.apply(lambda row: getDayOfWeek(row['start_month'], row['start_day'], row['start_year']), axis=1)
    taxi['season'] = taxi['start_month'].apply(getSeason)
    taxi['timeOfDay'] = taxi.apply(lambda row: getTimeofDay(row['start_hour']), axis=1)
    taxi['is_weekend'] = taxi['dayOfWeek'].apply(isWeekend)
    taxi['additionalStop'] = (taxi['NumberOfPassengers'] > 1).astype(int)
    taxi['is_holiday'] = taxi.apply(lambda row: isHoliday(row['start_month'], row['start_day'], row['start_year']), axis=1)
    taxi['minutes_since_midnight'] = taxi['start_hour'] * 60 + taxi['start_minute']
    taxi['minutes_until_midnight'] = 24 * 60 - taxi['minutes_since_midnight']

    # Distance-based values
    taxi['x2xDistance'] = abs(taxi['dropoff_x'] - taxi['pickup_x'])
    taxi['y2yDistance'] = abs(taxi['dropoff_y'] - taxi['pickup_y'])
    taxi['manhattan_distance'] = taxi['x2xDistance'] + taxi['y2yDistance']
    taxi['manhattan_distance_squared'] = taxi['manhattan_distance'] ** 2
    taxi['euclidean_distance'] = np.sqrt(taxi['x2xDistance'] ** 2 + taxi['y2yDistance'] ** 2)
    taxi['euclidean_distance_squared'] = taxi['euclidean_distance'] ** 2
    taxi['distance_category'] = pd.cut(taxi['euclidean_distance'], bins=[0, 1, 2, 5, 10, float('inf')], labels=['very_short', 'short', 'medium', 'long', 'very_long']).cat.codes

        # Relating distances to potentially related metrics
    taxi['distance_x_timeOfDay'] = taxi['x2xDistance'] * taxi['timeOfDay']
    taxi['distance_y_timeOfDay'] = taxi['y2yDistance'] * taxi['timeOfDay']
    taxi['distance_x_weekend'] = taxi['x2xDistance'] * taxi['is_weekend']
    taxi['distance_y_weekend'] = taxi['y2yDistance'] * taxi['is_weekend']
    taxi['distance_x_hour'] = taxi['x2xDistance'] * taxi['start_hour']
    taxi['distance_y_hour'] = taxi['y2yDistance'] * taxi['start_hour']
    taxi['distance_x_dest_x'] = taxi['x2xDistance'] * taxi['dropoff_x']
    taxi['distance_y_dest_y'] = taxi['y2yDistance'] * taxi['dropoff_y']

        # Adding non-linearity
    taxi['x2xDistance_squared'] = taxi['x2xDistance'] ** 2
    taxi['y2yDistance_squared'] = taxi['y2yDistance'] ** 2
    taxi['x2xDistance_cubed'] = taxi['x2xDistance'] ** 3
    taxi['y2yDistance_cubed'] = taxi['y2yDistance'] ** 3

    # Time-based metrics
    taxi['hour_sin'] = np.sin(2 * np.pi * taxi['start_hour'] / 24)
    taxi['hour_cos'] = np.cos(2 * np.pi * taxi['start_hour'] / 24)
    taxi['month_sin'] = np.sin(2 * np.pi * taxi['start_month'] / 12)
    taxi['month_cos'] = np.cos(2 * np.pi * taxi['start_month'] / 12)
    taxi['dayOfWeek_sin'] = np.sin(2 * np.pi * taxi['dayOfWeek'] / 7)
    taxi['dayOfWeek_cos'] = np.cos(2 * np.pi * taxi['dayOfWeek'] / 7)
    taxi['is_night'] = ((taxi['start_hour'] >= 22) | (taxi['start_hour'] < 5)).astype(int)

    taxi['hour_x_is_weekend'] = taxi['start_hour'] * taxi['is_weekend']
    taxi['timeOfDay_x_season'] = taxi['timeOfDay'] * taxi['season']
    taxi['is_rush_hour'] = (((taxi['start_hour'] >= 8) & (taxi['start_hour'] <= 10)) | ((taxi['start_hour'] >= 17) & (taxi['start_hour'] <= 19))).astype(int)

    # Passenger values
    taxi['solo_passenger'] = (taxi['NumberOfPassengers'] == 1).astype(int)
    taxi['many_passengers'] = (taxi['NumberOfPassengers'] > 3).astype(int)

    # Spacial values
    taxi['pickup_quadrant_x'] = (taxi['pickup_x'] > taxi['pickup_x'].median()).astype(int)
    taxi['pickup_quadrant_y'] = (taxi['pickup_y'] > taxi['pickup_y'].median()).astype(int)
    taxi['dropoff_quadrant_x'] = (taxi['dropoff_x'] > taxi['dropoff_x'].median()).astype(int)
    taxi['dropoff_quadrant_y'] = (taxi['dropoff_y'] > taxi['dropoff_y'].median()).astype(int)
    
    taxi['cross_quadrant_trip'] = (
        (taxi['pickup_quadrant_x'] != taxi['dropoff_quadrant_x']) |
        (taxi['pickup_quadrant_y'] != taxi['dropoff_quadrant_y'])
    ).astype(int)

    return taxi

# ============== Linear Regression ==============
features = [
    'start_hour', 'start_minute', 'start_day', 'start_month', 
    'x2xDistance', 'dayOfWeek', 
    'additionalStop', 'is_weekend', 
    'x2xDistance_squared', 'y2yDistance_squared', 
    'x2xDistance_cubed', 'y2yDistance_cubed',
    'distance_x_timeOfDay', 'distance_y_timeOfDay', 
    'distance_x_weekend', 'distance_y_weekend',
    'distance_x_hour', 'distance_y_hour',
    'hour_sin', 'hour_cos', 'month_sin', 
    'dayOfWeek_sin', 'dayOfWeek_cos',
    'manhattan_distance_squared', 'euclidean_distance',
    'distance_category',
    'distance_x_dest_x', 'distance_y_dest_y',
    'hour_x_is_weekend', 'timeOfDay_x_season'
]

print('==============|Preparing Data|==============')

df = pd.read_csv('Train.csv')

taxi = setup_features(df)
scaler = StandardScaler()
x_raw = taxi[features]
x = scaler.fit_transform(x_raw)

y = taxi['duration']

print('==============|Data Ready|==============')

if ENABLE_LINEAR_REGRESSION:
    print('==============|Linear Regression|==============')
    
    # Linear Regression with 5-fold cross-validation
    linear_model = LinearRegression()
    linear_cv_scores = cross_val_score(linear_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')

    print(f"Linear Mean CV R²: {linear_cv_scores.mean():.6f}")
    print(f"Linear Std Dev CV R²: {linear_cv_scores.std():.6f}")

if ENABLE_RIDGE_REGRESSION:
    print('==============|Ridge Regression|==============')
    ridge_alphas = np.logspace(-3, 4, 150)  # Expanded range: 0.001 to 10000
    ridge_model = Ridge()

    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    ridge_grid = GridSearchCV(ridge_pipeline, {'ridge__alpha': ridge_alphas}, cv=5, scoring='r2', n_jobs=12)
    ridge_grid.fit(x, y)

    print(f"Best Ridge Alpha: {ridge_grid.best_params_['ridge__alpha']:.6f}")
    print(f"Best Ridge CV R² Score: {ridge_grid.best_score_:.6f}")

    # Evaluate best Ridge model with k-fold
    ridge_best = Ridge(alpha=ridge_grid.best_params_['ridge__alpha'])
    ridge_cv_scores = cross_val_score(ridge_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"Ridge CV R² Scores (5 folds): {ridge_cv_scores}")
    print(f"Ridge Mean CV R²: {ridge_cv_scores.mean():.6f}")
    print(f"Ridge Std Dev CV R²: {ridge_cv_scores.std():.6f}")

if ENABLE_LASSO_REGRESSION:
    print('==============|Lasso Regression|==============')
    
    # Use LassoCV for efficient alpha selection (built-in cross-validation)
    # This is much faster than GridSearchCV with many alphas
    lasso_cv = LassoCV(
        alphas=np.logspace(-4, 2, 50),  # 50 alpha values (fast!)
        cv=5,
        max_iter=5000,
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
        verbose=0
    )
    
    lasso_cv.fit(x, y)
    
    print(f"Best Lasso Alpha: {lasso_cv.alpha_:.6f}")
    print(f"Lasso CV R² Score: {lasso_cv.score(x, y):.6f}")
    
    # Evaluate best Lasso model with k-fold
    lasso_best = Lasso(alpha=lasso_cv.alpha_, max_iter=5000)
    lasso_cv_scores = cross_val_score(lasso_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"Lasso CV R² Scores (5 folds): {lasso_cv_scores}")
    print(f"Lasso Mean CV R²: {lasso_cv_scores.mean():.6f}")
    print(f"Lasso Std Dev CV R²: {lasso_cv_scores.std():.6f}")

if ENABLE_NEURAL_NETWORK:
    print('==============|Neural Network|==============')
    
    # Using cuda, will need to change if not using Nvidia GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train_nn_model(X_train_scaled, y_train, X_val_scaled, y_val, 
                       hidden_layers, learning_rate, dropout_rate, batch_size, device):
        class ConfigurableNN(nn.Module):
            def __init__(self, input_size, hidden_layers, dropout_rate):
                super(ConfigurableNN, self).__init__()
                
                layer_sizes = [input_size] + hidden_layers + [1]
                self.layers = nn.ModuleList()
                self.batch_norms = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                
                for i in range(len(layer_sizes) - 2):
                    self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                    self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
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
        
        model = ConfigurableNN(X_train_scaled.shape[1], hidden_layers, dropout_rate).to(device)
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
        
        for epoch in range(100):
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
                if patience_counter >= 15:
                    model.load_state_dict(best_model_state)
                    break
        
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val_tensor).cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()
        
        ss_res = np.sum((y_val_np - y_pred_val) ** 2)
        ss_tot = np.sum((y_val_np - np.mean(y_val_np)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return r2
    
    def nn_objective(trial):
        # Suggest hyperparameters
        n_layers = trial.suggest_int('n_layers', 2, 3)
        hidden_layers = [trial.suggest_int(f'hidden_size_{i}', 32, 256) for i in range(n_layers)]
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
        batch_size = trial.suggest_int('batch_size', 16, 64)
        
        # Evaluate with 3-fold CV (faster)
        cv_scores = []
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(x)):
            X_train_fold = x[train_idx]
            X_val_fold = x[val_idx]
            y_train_fold = y.values[train_idx]
            y_val_fold = y.values[val_idx]
            
            r2 = train_nn_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                               hidden_layers, learning_rate, dropout_rate, batch_size, device)
            cv_scores.append(r2)
        
        return np.mean(cv_scores)
    
    # Optuna optimization
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(nn_objective, n_trials=15, show_progress_bar=True)
    
    best_nn_trial = study.best_trial
    best_hidden_layers = [best_nn_trial.params[f'hidden_size_{i}'] 
                          for i in range(best_nn_trial.params['n_layers'])]
    
    print(f"\nBest Neural Network Trial:")
    print(f"  Mean CV R²: {best_nn_trial.value:.6f}\n")

if ENABLE_RANDOM_FOREST:
    print('==============|Random Forest|==============\n')

    def rf_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 10, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=12
        )
        
        # 3-fold CV for speed
        cv_scores = cross_val_score(rf_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), 
                                   scoring='r2', n_jobs=12)
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(rf_objective, n_trials=20, show_progress_bar=True)
    
    best_rf_trial = study.best_trial
    
    print(f"\nBest Random Forest Trial:")
    for param, value in best_rf_trial.params.items():
        print(f"  {param}: {value}")
    print(f"\nBest Random Forest Mean CV R²: {best_rf_trial.value:.6f}")

    # Evaluate best Random Forest model with 5-fold CV (final evaluation)
    rf_best = RandomForestRegressor(
        n_estimators=best_rf_trial.params['n_estimators'],
        max_depth=best_rf_trial.params['max_depth'],
        min_samples_split=best_rf_trial.params['min_samples_split'],
        min_samples_leaf=best_rf_trial.params['min_samples_leaf'],
        max_features=best_rf_trial.params['max_features'],
        random_state=42,
        n_jobs=12
    )

    rf_cv_scores = cross_val_score(rf_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=12)
    print(f"\nRandom Forest CV R² Scores (5 folds): {rf_cv_scores}")
    print(f"Random Forest Mean CV R²: {rf_cv_scores.mean():.6f}")
    print(f"Random Forest Std Dev CV R²: {rf_cv_scores.std():.6f}")

if ENABLE_DECISION_TREE:
    print('==============|Decision Tree|==============\n')

    def dt_objective(trial):
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)
        splitter = trial.suggest_categorical('splitter', ['best', 'random'])
        
        dt_model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            splitter=splitter,
            random_state=42
        )
        
        cv_scores = cross_val_score(dt_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), 
                                   scoring='r2')
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(dt_objective, n_trials=15, show_progress_bar=True)
    
    best_dt_trial = study.best_trial
    
    print(f"\nBest Decision Tree Trial:")
    for param, value in best_dt_trial.params.items():
        print(f"  {param}: {value}")
    print(f"\nBest Decision Tree Mean CV R²: {best_dt_trial.value:.6f}")

    dt_best = DecisionTreeRegressor(
        max_depth=best_dt_trial.params['max_depth'],
        min_samples_split=best_dt_trial.params['min_samples_split'],
        min_samples_leaf=best_dt_trial.params['min_samples_leaf'],
        splitter=best_dt_trial.params['splitter'],
        random_state=42
    )

    dt_cv_scores = cross_val_score(dt_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"\nDecision Tree CV R² Scores (5 folds): {dt_cv_scores}")
    print(f"Decision Tree Mean CV R²: {dt_cv_scores.mean():.6f}")
    print(f"Decision Tree Std Dev CV R²: {dt_cv_scores.std():.6f}\n")

if ENABLE_GRADIENT_BOOSTING:
    print('==============|Gradient Boosting|==============\n')

    def gb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        
        gb_model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=42
        )
        
        cv_scores = cross_val_score(gb_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), 
                                   scoring='r2')
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(gb_objective, n_trials=15, show_progress_bar=True)
    
    best_gb_trial = study.best_trial
    
    print(f"\nBest Gradient Boosting Trial:")
    for param, value in best_gb_trial.params.items():
        print(f"  {param}: {value}")
    print(f"\nBest Gradient Boosting Mean CV R²: {best_gb_trial.value:.6f}")

    gb_best = GradientBoostingRegressor(
        n_estimators=best_gb_trial.params['n_estimators'],
        learning_rate=best_gb_trial.params['learning_rate'],
        max_depth=best_gb_trial.params['max_depth'],
        min_samples_split=best_gb_trial.params['min_samples_split'],
        min_samples_leaf=best_gb_trial.params['min_samples_leaf'],
        subsample=best_gb_trial.params['subsample'],
        random_state=42
    )

    gb_cv_scores = cross_val_score(gb_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"\nGradient Boosting CV R² Scores (5 folds): {gb_cv_scores}")
    print(f"Gradient Boosting Mean CV R²: {gb_cv_scores.mean():.6f}")
    print(f"Gradient Boosting Std Dev CV R²: {gb_cv_scores.std():.6f}\n")

if ENABLE_ADABOOST:
    print('==============|AdaBoost|==============\n')

    def ab_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 1.0, log=True)
        loss = trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
        
        ab_model = __import__('sklearn.ensemble', fromlist=['AdaBoostRegressor']).AdaBoostRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=42
        )
        
        cv_scores = cross_val_score(ab_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), 
                                   scoring='r2')
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(ab_objective, n_trials=15, show_progress_bar=True)
    
    best_ab_trial = study.best_trial
    
    print(f"\nBest AdaBoost Trial:")
    for param, value in best_ab_trial.params.items():
        print(f"  {param}: {value}")
    print(f"\nBest AdaBoost Mean CV R²: {best_ab_trial.value:.6f}")

    ab_best = __import__('sklearn.ensemble', fromlist=['AdaBoostRegressor']).AdaBoostRegressor(
        n_estimators=best_ab_trial.params['n_estimators'],
        learning_rate=best_ab_trial.params['learning_rate'],
        loss=best_ab_trial.params['loss'],
        random_state=42
    )

    ab_cv_scores = cross_val_score(ab_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"\nAdaBoost CV R² Scores (5 folds): {ab_cv_scores}")
    print(f"AdaBoost Mean CV R²: {ab_cv_scores.mean():.6f}")
    print(f"AdaBoost Std Dev CV R²: {ab_cv_scores.std():.6f}\n")

if ENABLE_XGBOOST:
    print('==============|XGBoost|==============\n')

    def xgb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 7)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
            n_jobs=12,
            verbosity=0
        )
        
        cv_scores = cross_val_score(xgb_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), 
                                   scoring='r2', n_jobs=12)
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(xgb_objective, n_trials=15, show_progress_bar=True)
    
    best_xgb_trial = study.best_trial
    
    print(f"\nBest XGBoost Trial:")
    for param, value in best_xgb_trial.params.items():
        print(f"  {param}: {value}")
    print(f"\nBest XGBoost Mean CV R²: {best_xgb_trial.value:.6f}")

    xgb_best = xgb.XGBRegressor(
        n_estimators=best_xgb_trial.params['n_estimators'],
        learning_rate=best_xgb_trial.params['learning_rate'],
        max_depth=best_xgb_trial.params['max_depth'],
        min_child_weight=best_xgb_trial.params['min_child_weight'],
        subsample=best_xgb_trial.params['subsample'],
        colsample_bytree=best_xgb_trial.params['colsample_bytree'],
        random_state=42,
        n_jobs=12,
        verbosity=0
    )

    xgb_cv_scores = cross_val_score(xgb_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=12)
    print(f"\nXGBoost CV R² Scores (5 folds): {xgb_cv_scores}")
    print(f"XGBoost Mean CV R²: {xgb_cv_scores.mean():.6f}")
    print(f"XGBoost Std Dev CV R²: {xgb_cv_scores.std():.6f}\n")

if ENABLE_CATBOOST:
    print('==============|CatBoost|==============\n')

    def cb_objective(trial):
        iterations = trial.suggest_int('iterations', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
        depth = trial.suggest_int('depth', 4, 10)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1, 10)
        
        cb_model = __import__('catboost', fromlist=['CatBoostRegressor']).CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_state=42,
            verbose=False
        )
        
        cv_scores = cross_val_score(cb_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), 
                                   scoring='r2')
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(cb_objective, n_trials=15, show_progress_bar=True)
    
    best_cb_trial = study.best_trial
    
    print(f"\nBest CatBoost Trial:")
    for param, value in best_cb_trial.params.items():
        print(f"  {param}: {value}")
    print(f"\nBest CatBoost Mean CV R²: {best_cb_trial.value:.6f}")

    cb_best = __import__('catboost', fromlist=['CatBoostRegressor']).CatBoostRegressor(
        iterations=best_cb_trial.params['iterations'],
        learning_rate=best_cb_trial.params['learning_rate'],
        depth=best_cb_trial.params['depth'],
        l2_leaf_reg=best_cb_trial.params['l2_leaf_reg'],
        random_state=42,
        verbose=False
    )

    cb_cv_scores = cross_val_score(cb_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"\nCatBoost CV R² Scores (5 folds): {cb_cv_scores}")
    print(f"CatBoost Mean CV R²: {cb_cv_scores.mean():.6f}")
    print(f"CatBoost Std Dev CV R²: {cb_cv_scores.std():.6f}\n")

if ENABLE_LIGHTGBM:
    print('==============|LightGBM|==============\n')

    def lgb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
        num_leaves = trial.suggest_int('num_leaves', 20, 100)
        min_child_samples = trial.suggest_int('min_child_samples', 5, 50)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            random_state=42,
            n_jobs=12,
            verbose=-1
        )
        
        cv_scores = cross_val_score(lgb_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), 
                                   scoring='r2', n_jobs=12)
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(lgb_objective, n_trials=15, show_progress_bar=True)
    
    best_lgb_trial = study.best_trial
    
    print(f"\nBest LightGBM Trial:")
    for param, value in best_lgb_trial.params.items():
        print(f"  {param}: {value}")
    print(f"\nBest LightGBM Mean CV R²: {best_lgb_trial.value:.6f}")

    lgb_best = lgb.LGBMRegressor(
        n_estimators=best_lgb_trial.params['n_estimators'],
        learning_rate=best_lgb_trial.params['learning_rate'],
        num_leaves=best_lgb_trial.params['num_leaves'],
        min_child_samples=best_lgb_trial.params['min_child_samples'],
        subsample=best_lgb_trial.params['subsample'],
        random_state=42,
        n_jobs=12,
        verbose=-1
    )

    lgb_cv_scores = cross_val_score(lgb_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=12)
    print(f"\nLightGBM CV R² Scores (5 folds): {lgb_cv_scores}")
    print(f"LightGBM Mean CV R²: {lgb_cv_scores.mean():.6f}")
    print(f"LightGBM Std Dev CV R²: {lgb_cv_scores.std():.6f}\n")


# Cross Validation Curves of top 3 R^2 models

