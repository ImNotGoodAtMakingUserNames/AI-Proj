import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
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

import matplotlib.pyplot as plt
    

sample_frac = 0.3
# ============== MODEL TOGGLES ==============
ENABLE_LINEAR_REGRESSION = False
ENABLE_RIDGE_REGRESSION = False
ENABLE_LASSO_REGRESSION = False
ENABLE_NEURAL_NETWORK = False

    # Tree-based models
ENABLE_RANDOM_FOREST = False
ENABLE_DECISION_TREE = False
ENABLE_GRADIENT_BOOSTING = False
ENABLE_ADABOOST = False
ENABLE_XGBOOST = False
ENABLE_CATBOOST = False
ENABLE_LIGHTGBM = False

# Advanced sections
ENABLE_CV_CURVES = True
ENABLE_FULL_TRAINING = False
ENABLE_FEATURE_SELECTION = True  # Set to True to use only top 30 features
ENABLE_FEATURE_REFINE = True
# ============================================

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

def setup_features(df, sample_fraction=None):
    """Setup features for the dataset. If sample_fraction is None, uses global sample_frac"""
    if sample_fraction is None:
        sample_fraction = sample_frac
        
# Read Data
    taxi = df.sample(frac=sample_fraction, random_state=42).copy()

    print('==============|Training Data Imported|==============')
    print(f'Using {sample_fraction*100}% of training data ({len(taxi)} rows out of {len(df)} total rows)\n')

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
    taxi['euclidean_distance_cubed'] = taxi['euclidean_distance'] ** 3
    taxi['distance_category'] = pd.cut(taxi['euclidean_distance'], bins=[0, 0.5, 1, 2, 5, 10, float('inf')], labels=['very_short', 'short', 'medium', 'long', 'very_long', 'very_long_plus']).cat.codes

        # Relating distances to potentially related metrics
    taxi['distance_x_timeOfDay'] = taxi['x2xDistance'] * taxi['timeOfDay']
    taxi['distance_y_timeOfDay'] = taxi['y2yDistance'] * taxi['timeOfDay']
    taxi['distance_x_weekend'] = taxi['x2xDistance'] * taxi['is_weekend']
    taxi['distance_y_weekend'] = taxi['y2yDistance'] * taxi['is_weekend']
    taxi['distance_x_hour'] = taxi['x2xDistance'] * taxi['start_hour']
    taxi['distance_x_hour_squared'] = taxi['distance_x_hour'] ** 2
    taxi['distance_y_hour'] = taxi['y2yDistance'] * taxi['start_hour']
    taxi['distance_x_dest_x'] = taxi['x2xDistance'] * taxi['dropoff_x']
    taxi['distance_y_dest_y'] = taxi['y2yDistance'] * taxi['dropoff_y']
    taxi['distance_x_passengers'] = taxi['euclidean_distance'] * taxi['NumberOfPassengers']

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
    taxi['passengers_x_weekend'] = taxi['NumberOfPassengers'] * taxi['is_weekend']
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

print('==============|Preparing Data|==============')

df = pd.read_csv('Train.csv')

taxi = setup_features(df)
x_raw = taxi[features]
x = x_raw  # Use raw features (tree models don't benefit from scaling)

# Create scaled version for linear models and neural networks
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_raw)

y = taxi['duration']

print('==============|Data Ready|==============')

# Feature selection phase: Use top 30 features if requested
selected_features = features  # Default to all features
top_30_features = None

if ENABLE_FEATURE_SELECTION or ENABLE_LINEAR_REGRESSION:
    print('\n==============|Feature Selection Phase|==============\n')
    
    # Train quick linear regression to identify top features
    print("Identifying top features using linear regression...")
    linear_selector = LinearRegression()
    linear_selector.fit(x_scaled, y)
    
    # Get feature importances (absolute coefficients)
    feature_importance = np.abs(linear_selector.coef_)
    
    # Create DataFrame with features and their importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': linear_selector.coef_,
        'Abs_Coefficient': feature_importance
    }).sort_values('Abs_Coefficient', ascending=False)
    
    # Select top 30 features
    top_30_features = importance_df.head(30)['Feature'].tolist()
    top_30_indices = [features.index(f) for f in top_30_features]
    
    print(f"\n==============|Top 30 Most Important Features|==============\n")
    print(importance_df.head(30)[['Feature', 'Coefficient', 'Abs_Coefficient']].to_string(index=False))
    
    selected_features = top_30_features
    x = x[:, top_30_indices]
    x_scaled = x_scaled[:, top_30_indices]
    print(f"\n*** Using TOP 30 FEATURES for all model evaluations ***\n")

if ENABLE_LINEAR_REGRESSION:
    print('==============|Linear Regression|==============')
    
    # Linear Regression with 5-fold cross-validation
    linear_model = LinearRegression()
    linear_cv_scores = cross_val_score(linear_model, x_scaled, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')

    print(f"Linear Mean CV R²: {linear_cv_scores.mean():.6f}\n")

if ENABLE_RIDGE_REGRESSION:
    print('==============|Ridge Regression|==============')
    ridge_alphas = np.logspace(-3, 4, 150)  # Expanded range: 0.001 to 10000
    ridge_model = Ridge()

    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    ridge_grid = GridSearchCV(ridge_pipeline, {'ridge__alpha': ridge_alphas}, cv=5, scoring='r2', n_jobs=12)
    ridge_grid.fit(x_scaled, y)

    print(f"Best Ridge Alpha: {ridge_grid.best_params_['ridge__alpha']:.6f}")
    print(f"Best Ridge CV R² Score: {ridge_grid.best_score_:.6f}")

    # Evaluate best Ridge model with k-fold
    ridge_best = Ridge(alpha=ridge_grid.best_params_['ridge__alpha'])
    ridge_cv_scores = cross_val_score(ridge_best, x_scaled, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"Ridge Mean CV R²: {ridge_cv_scores.mean():.6f}")

if ENABLE_LASSO_REGRESSION:
    print('==============|Lasso Regression|==============')
    
    # LassoCV for optomization 
    lasso_cv = LassoCV(
        alphas=np.logspace(-4, 2, 50),  # 50 alpha values (fast!)
        cv=5,
        max_iter=5000,
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
        verbose=0
    )
    
    lasso_cv.fit(x_scaled, y)
    
    print(f"Best Lasso Alpha: {lasso_cv.alpha_:.6f}")
    print(f"Lasso CV R² Score: {lasso_cv.score(x_scaled, y):.6f}")
    
    # Evaluate best Lasso model with k-fold
    lasso_best = Lasso(alpha=lasso_cv.alpha_, max_iter=5000)
    lasso_cv_scores = cross_val_score(lasso_best, x_scaled, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
    print(f"Lasso Mean CV R²: {lasso_cv_scores.mean():.6f}")

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
        
        # Evaluate with 3-fold CV to save on time
        cv_scores = []
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(x_scaled)):
            X_train_fold = x_scaled[train_idx]
            X_val_fold = x_scaled[val_idx]
            y_train_fold = y.values[train_idx]
            y_val_fold = y.values[val_idx]
            
            r2 = train_nn_model(X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                               hidden_layers, learning_rate, dropout_rate, batch_size, device)
            cv_scores.append(r2)
        
        return np.mean(cv_scores)
    
    # Optuna optimization
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(nn_objective, n_trials=3, show_progress_bar=True)
    
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
        
        cv_scores = cross_val_score(rf_model, x, y, cv=KFold(n_splits=3, shuffle=True, random_state=42), 
                                   scoring='r2', n_jobs=12)
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(rf_objective, n_trials=10, show_progress_bar=True)
    
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
    print(f"\nRandom Forest Mean CV R²: {rf_cv_scores.mean():.6f}")

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
        
        cv_scores = cross_val_score(dt_model, x, y, cv=KFold(n_splits=3, shuffle=True, random_state=42), 
                                   scoring='r2')
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(dt_objective, n_trials=10, show_progress_bar=True)
    
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
    print(f"\nDecision Tree Mean CV R²: {dt_cv_scores.mean():.6f}\n")

if ENABLE_GRADIENT_BOOSTING:
    print('==============|Gradient Boosting|==============\n')

    def gb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 15)
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
        
        cv_scores = cross_val_score(gb_model, x, y, cv=KFold(n_splits=3, shuffle=True, random_state=42), 
                                   scoring='r2')
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(gb_objective, n_trials=10, show_progress_bar=True)
    
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
    print(f"\nGradient Boosting Mean CV R²: {gb_cv_scores.mean():.6f}\n")

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
        
        cv_scores = cross_val_score(ab_model, x, y, cv=KFold(n_splits=3, shuffle=True, random_state=42), 
                                   scoring='r2')
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(ab_objective, n_trials=10, show_progress_bar=True)
    
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
    print(f"\nAdaBoost Mean CV R²: {ab_cv_scores.mean():.6f}\n")

if ENABLE_XGBOOST:
    print('==============|XGBoost|==============\n')

    def xgb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 7)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        gamma = trial.suggest_float('gamma', 0, 5)
        reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
        reg_lambda = trial.suggest_float('reg_lambda', 0, 10)
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=12,
            verbosity=0
        )
        
        cv_scores = cross_val_score(xgb_model, x, y, cv=KFold(n_splits=3, shuffle=True, random_state=42), 
                                   scoring='r2', n_jobs=12)
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(xgb_objective, n_trials=10, show_progress_bar=True)
    
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
        gamma=best_xgb_trial.params.get('gamma', 0),
        reg_alpha=best_xgb_trial.params.get('reg_alpha', 0),
        reg_lambda=best_xgb_trial.params.get('reg_lambda', 1),
        random_state=42,
        n_jobs=12,
        verbosity=0
    )

    xgb_cv_scores = cross_val_score(xgb_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=12)
    print(f"\nXGBoost Mean CV R²: {xgb_cv_scores.mean():.6f}\n")
    
    # Save best parameters to JSON
    xgb_params_file = 'results_logs/best_xgboost_params.json'
    with open(xgb_params_file, 'w') as f:
        json.dump({'params': best_xgb_trial.params, 'r2_score': best_xgb_trial.value}, f, indent=2)
    print(f"Saved XGBoost best parameters to {xgb_params_file}\n")

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
        
        cv_scores = cross_val_score(cb_model, x, y, cv=KFold(n_splits=3, shuffle=True, random_state=42), 
                                   scoring='r2')
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(cb_objective, n_trials=10, show_progress_bar=True)
    
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
    print(f"\nCatBoost Mean CV R²: {cb_cv_scores.mean():.6f}\n")

if ENABLE_LIGHTGBM:
    print('==============|LightGBM|==============\n')

    def lgb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)
        num_leaves = trial.suggest_int('num_leaves', 20, 100)
        min_child_samples = trial.suggest_int('min_child_samples', 5, 50)
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        feature_fraction = trial.suggest_float('feature_fraction', 0.5, 1.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
        reg_lambda = trial.suggest_float('reg_lambda', 0, 10)
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            feature_fraction=feature_fraction,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=12,
            verbose=-1
        )
        
        cv_scores = cross_val_score(lgb_model, x, y, cv=KFold(n_splits=3, shuffle=True, random_state=42), 
                                   scoring='r2', n_jobs=12)
        return cv_scores.mean()
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(lgb_objective, n_trials=10, show_progress_bar=True)
    
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
        feature_fraction=best_lgb_trial.params.get('feature_fraction', 1.0),
        reg_alpha=best_lgb_trial.params.get('reg_alpha', 0),
        reg_lambda=best_lgb_trial.params.get('reg_lambda', 1),
        random_state=42,
        n_jobs=12,
        verbose=-1
    )

    lgb_cv_scores = cross_val_score(lgb_best, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=12)
    print(f"\nLightGBM Mean CV R²: {lgb_cv_scores.mean():.6f}\n")
    
    # Save best parameters to JSON
    lgb_params_file = 'results_logs/best_lightgbm_params.json'
    with open(lgb_params_file, 'w') as f:
        json.dump({'params': best_lgb_trial.params, 'r2_score': best_lgb_trial.value}, f, indent=2)
    print(f"Saved LightGBM best parameters to {lgb_params_file}\n")


# Store pre-computed best trial values from previous runs
class BestTrialMock:
    def __init__(self, params_dict, value=None):
        self.params = params_dict
        self.value = value

def load_best_params(model_name, default_params):
    """Load best parameters from JSON file, with fallback to defaults"""
    params_file = f'results_logs/best_{model_name}_params.json'
    
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f:
                data = json.load(f)
                params = data.get('params', default_params)
                r2_score = data.get('r2_score', None)
                print(f"Loaded {model_name} best parameters from {params_file}")
                if r2_score:
                    print(f"  Previous best R² score: {r2_score:.6f}")
                return BestTrialMock(params, r2_score)
        except Exception as e:
            print(f"Failed to load {model_name} parameters from {params_file}: {e}")
            print(f"  Using default fallback parameters")
            return BestTrialMock(default_params)
    else:
        print(f"{params_file} not found, using default fallback parameters")
        return BestTrialMock(default_params)

# Load best parameters from JSON or use defaults
best_xgb_trial = load_best_params('xgboost', {
    'n_estimators': 250,
    'learning_rate': 0.1,
    'max_depth': 11,
    'min_child_weight': 5,
    'subsample': 1.0,
    'colsample_bytree': 0.9,
    'gamma': 0,
    'reg_alpha': 0.1,
    'reg_lambda': 1
})

best_lgb_trial = load_best_params('lightgbm', {
    'n_estimators': 144,
    'learning_rate': 0.22648248189516848,
    'num_leaves': 79,
    'min_child_samples': 32,
    'subsample': 0.6624074561769746,
    'feature_fraction': 1.0,
    'reg_alpha': 0,
    'reg_lambda': 1
})

# Support for other models (not used in full training, but kept for completeness)
best_rf_trial = BestTrialMock({
    'n_estimators': 271,
    'max_depth': 24,
    'min_samples_split': 7,
    'min_samples_leaf': 3,
    'max_features': 'sqrt'
})

best_dt_trial = BestTrialMock({
    'max_depth': 9,
    'min_samples_split': 7,
    'min_samples_leaf': 5,
    'splitter': 'best'
})

best_gb_trial = BestTrialMock({
    'n_estimators': 144,
    'learning_rate': 0.07969454818643935,
    'max_depth': 8,
    'min_samples_split': 13,
    'min_samples_leaf': 2,
    'subsample': 0.662397808134481
})

best_cb_trial = BestTrialMock({
    'iterations': 144,
    'learning_rate': 0.22648248189516848,
    'depth': 9,
    'l2_leaf_reg': 6.387926357773329
})

# Cross Validation Curves of top 4 R^2 models
if ENABLE_CV_CURVES:
    print('==============|Cross-Validation Curves: Depth Analysis|==============\n')



    # Define depth range
    depths = list(range(1, 15))  # 2 to 17 inclusive

    # Checkpoint file for resuming work
    checkpoint_file = 'results_logs/cv_curves_checkpoint.json'

    # Load checkpoint if it exists
    checkpoint_data = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded checkpoint from {checkpoint_file}\n")

    # Store CV scores for each model
    xgb_depths_scores = checkpoint_data.get('xgb_depths_scores', [])
    catboost_depths_scores = checkpoint_data.get('catboost_depths_scores', [])
    lightgbm_depths_scores = checkpoint_data.get('lightgbm_depths_scores', [])
    gb_depths_scores = checkpoint_data.get('gb_depths_scores', [])

    # XGBoost across depths
    if len(xgb_depths_scores) == 0:
        print("XGBoost: ")
        for depth in depths:
            xgb_model = xgb.XGBRegressor(
                n_estimators=best_xgb_trial.params['n_estimators'],
                learning_rate=best_xgb_trial.params['learning_rate'],
                max_depth=depth,
                min_child_weight=best_xgb_trial.params['min_child_weight'],
                subsample=best_xgb_trial.params['subsample'],
                colsample_bytree=best_xgb_trial.params['colsample_bytree'],
                gamma=best_xgb_trial.params.get('gamma', 0),
                reg_alpha=best_xgb_trial.params.get('reg_alpha', 0),
                reg_lambda=best_xgb_trial.params.get('reg_lambda', 1),
                random_state=42,
                n_jobs=12,
                verbosity=0
            )
            cv_scores = cross_val_score(xgb_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=12)
            mean_score = cv_scores.mean()
            xgb_depths_scores.append(mean_score)
            print(f"  Depth {depth:2d}: R² = {mean_score:.6f}")
        print(" Done")
        checkpoint_data['xgb_depths_scores'] = xgb_depths_scores
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded XGBoost results from checkpoint ({len(xgb_depths_scores)} depths)")
    # CatBoost across depths
    if len(catboost_depths_scores) == 0:
        print("CatBoost: ")
        for depth in depths:
            cb_model = __import__('catboost', fromlist=['CatBoostRegressor']).CatBoostRegressor(
                iterations=best_cb_trial.params['iterations'],
                learning_rate=best_cb_trial.params['learning_rate'],
                depth=depth,
                l2_leaf_reg=best_cb_trial.params['l2_leaf_reg'],
                random_state=42,
                verbose=False
            )
            cv_scores = cross_val_score(cb_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
            mean_score = cv_scores.mean()
            catboost_depths_scores.append(mean_score)
            print(f"  Depth {depth:2d}: R² = {mean_score:.6f}")
        print(" Done")
        checkpoint_data['catboost_depths_scores'] = catboost_depths_scores
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded CatBoost results from checkpoint ({len(catboost_depths_scores)} depths)")

    # LightGBM across depths
    if len(lightgbm_depths_scores) == 0:
        print("LightGBM: ")
        for depth in depths:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=best_lgb_trial.params['n_estimators'],
                learning_rate=best_lgb_trial.params['learning_rate'],
                num_leaves=2**depth,  # LightGBM uses num_leaves instead of max_depth
                min_child_samples=best_lgb_trial.params['min_child_samples'],
                subsample=best_lgb_trial.params['subsample'],
                feature_fraction=best_lgb_trial.params.get('feature_fraction', 1.0),
                reg_alpha=best_lgb_trial.params.get('reg_alpha', 0),
                reg_lambda=best_lgb_trial.params.get('reg_lambda', 1),
                random_state=42,
                n_jobs=12,
                verbose=-1
            )
            cv_scores = cross_val_score(lgb_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2', n_jobs=12)
            mean_score = cv_scores.mean()
            lightgbm_depths_scores.append(mean_score)
            print(f"  Depth {depth:2d}: R² = {mean_score:.6f}")
        print(" Done")
        checkpoint_data['lightgbm_depths_scores'] = lightgbm_depths_scores
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded LightGBM results from checkpoint ({len(lightgbm_depths_scores)} depths)")

    # Gradient Boosting across depths (LIMITED TO 3 DEPTHS)
    gb_depths_limited = [1, 2, 3]
    if len(gb_depths_scores) == 0:
        print("Gradient Boosting (LIMITED to 3 depths): ")
        for depth in gb_depths_limited:
            gb_model = GradientBoostingRegressor(
                n_estimators=best_gb_trial.params['n_estimators'],
                learning_rate=best_gb_trial.params['learning_rate'],
                max_depth=depth,
                min_samples_split=best_gb_trial.params['min_samples_split'],
                min_samples_leaf=best_gb_trial.params['min_samples_leaf'],
                subsample=best_gb_trial.params['subsample'],
                random_state=42
            )
            cv_scores = cross_val_score(gb_model, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='r2')
            mean_score = cv_scores.mean()
            gb_depths_scores.append(mean_score)
            print(f"  Depth {depth:2d}: R² = {mean_score:.6f}")
        print(" Done")
        checkpoint_data['gb_depths_scores'] = gb_depths_scores
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loaded Gradient Boosting results from checkpoint ({len(gb_depths_scores)} depths)")

    # Save checkpoint after all models complete
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint saved to {checkpoint_file}\n")

    # Plot the curves
    plt.figure(figsize=(12, 7))
    plt.plot(depths, xgb_depths_scores, marker='o', linewidth=2.5, label='XGBoost', color='#FF6B6B', markersize=6)
    plt.plot(depths, catboost_depths_scores, marker='s', linewidth=2.5, label='CatBoost', color='#4ECDC4', markersize=6)
    plt.plot(depths, lightgbm_depths_scores, marker='^', linewidth=2.5, label='LightGBM', color='#45B7D1', markersize=6)
    plt.plot(gb_depths_limited, gb_depths_scores, marker='D', linewidth=2.5, label='Gradient Boosting (3 depths)', color='#F7DC6F', markersize=6)

    plt.xlabel('Max Depth', fontsize=12, fontweight='bold')
    plt.ylabel('Mean CV R² Score (5-Fold)', fontsize=12, fontweight='bold')
    plt.title('Cross-Validation Curves: Boosting Models Across Depth Parameters', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(depths)
    plt.tight_layout()
    plt.savefig('results_logs/cv_curves_depth_comparison.png', dpi=300, bbox_inches='tight')
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Plot saved to results_logs/cv_curves_depth_comparison.png")
    plt.show()

    print("\n==============|Depth Analysis Summary|==============")
    print(f"XGBoost Best Depth: {depths[np.argmax(xgb_depths_scores)]} with R²: {max(xgb_depths_scores):.6f}")
    print(f"CatBoost Best Depth: {depths[np.argmax(catboost_depths_scores)]} with R²: {max(catboost_depths_scores):.6f}")
    print(f"LightGBM Best Depth: {depths[np.argmax(lightgbm_depths_scores)]} with R²: {max(lightgbm_depths_scores):.6f}")
    print(f"Gradient Boosting Best Depth: {gb_depths_limited[np.argmax(gb_depths_scores)]} with R²: {max(gb_depths_scores):.6f}")

if ENABLE_FULL_TRAINING:
    print("\n\n==============|Full 100% Data Training with Feature Selection|==============\n")
    
    # Reload full dataset with all features
    df_full = pd.read_csv('Train.csv')
    taxi_full = setup_features(df_full, sample_fraction=1.0)
    x_full_raw = taxi_full[features]
    scaler_full = StandardScaler()
    x_full = scaler_full.fit_transform(x_full_raw)
    y_full = taxi_full['duration']
    
    # Train XGBoost on full data
    print("Training XGBoost on 100% data...")
    xgb_full = xgb.XGBRegressor(
        n_estimators=best_xgb_trial.params['n_estimators'],
        learning_rate=best_xgb_trial.params['learning_rate'],
        max_depth=best_xgb_trial.params['max_depth'],
        min_child_weight=best_xgb_trial.params['min_child_weight'],
        subsample=best_xgb_trial.params['subsample'],
        colsample_bytree=best_xgb_trial.params['colsample_bytree'],
        gamma=best_xgb_trial.params.get('gamma', 0),
        reg_alpha=best_xgb_trial.params.get('reg_alpha', 0),
        reg_lambda=best_xgb_trial.params.get('reg_lambda', 1),
        random_state=42,
        n_jobs=12,
        verbosity=0
    )
    xgb_full.fit(x_full, y_full)
    xgb_full_pred = xgb_full.predict(x_full)
    xgb_full_r2 = r2_score(y_full, xgb_full_pred)
    xgb_full_rmse = np.sqrt(mean_squared_error(y_full, xgb_full_pred))
    print(f"XGBoost R² on full data: {xgb_full_r2:.6f}\n")
    print(f"XGBoost RMSE on full data: {xgb_full_rmse:.4f}\n")
    
    # Train LightGBM on full data
    print("Training LightGBM on 100% data...")
    lgb_full = lgb.LGBMRegressor(
        n_estimators=best_lgb_trial.params['n_estimators'],
        learning_rate=best_lgb_trial.params['learning_rate'],
        num_leaves=best_lgb_trial.params['num_leaves'],
        min_child_samples=best_lgb_trial.params['min_child_samples'],
        subsample=best_lgb_trial.params['subsample'],
        feature_fraction=best_lgb_trial.params.get('feature_fraction', 1.0),
        reg_alpha=best_lgb_trial.params.get('reg_alpha', 0),
        reg_lambda=best_lgb_trial.params.get('reg_lambda', 1),
        random_state=42,
        n_jobs=12,
        verbose=-1
    )
    lgb_full.fit(x_full, y_full)
    lgb_full_pred = lgb_full.predict(x_full)
    lgb_full_r2 = r2_score(y_full, lgb_full_pred)
    lgb_full_rmse = np.sqrt(mean_squared_error(y_full, lgb_full_pred))
    print(f"LightGBM R² on full data: {lgb_full_r2:.6f}\n")
    print(f"LightGBM RMSE on full data: {lgb_full_rmse:.4f}\n")
    
    # Extract feature importances (built-in to models)
    print("==============|Feature Importance Analysis|==============\n")
    
    xgb_importance = xgb_full.feature_importances_
    lgb_importance = lgb_full.feature_importances_
    
    # Create DataFrames for feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'XGBoost_Importance': xgb_importance,
        'LightGBM_Importance': lgb_importance,
    })
    
    # Normalize importances to 0-1 scale for easier comparison
    xgb_norm = xgb_importance / np.sum(xgb_importance)
    lgb_norm = lgb_importance / np.sum(lgb_importance)
    
    feature_importance_df['XGBoost_Normalized'] = xgb_norm
    feature_importance_df['LightGBM_Normalized'] = lgb_norm
    feature_importance_df['Average_Importance'] = (xgb_norm + lgb_norm) / 2
    
    # Sort by average importance
    feature_importance_df = feature_importance_df.sort_values('Average_Importance', ascending=False)
    
    # Display ALL features ranked by importance
    print("ALL Features Ranked by Average Importance:\n")
    print(feature_importance_df[['Feature', 'XGBoost_Normalized', 'LightGBM_Normalized', 'Average_Importance']].to_string(index=False))
    print()
    
    # Save feature importance to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    importance_file = f'results_logs/feature_importance_full_training_{timestamp}.csv'
    feature_importance_df.to_csv(importance_file, index=False)
    print(f"\nFull feature importance saved to: {importance_file}")
    
    # Save model hyperparameters
    hyperparams_file = f'results_logs/final_hyperparameters_full_training_{timestamp}.json'
    hyperparams = {
        'XGBoost': {
            'n_estimators': best_xgb_trial.params['n_estimators'],
            'learning_rate': best_xgb_trial.params['learning_rate'],
            'max_depth': best_xgb_trial.params['max_depth'],
            'min_child_weight': best_xgb_trial.params['min_child_weight'],
            'subsample': best_xgb_trial.params['subsample'],
            'colsample_bytree': best_xgb_trial.params['colsample_bytree'],
            'r2_score': float(xgb_full_r2)
        },
        'LightGBM': {
            'n_estimators': best_lgb_trial.params['n_estimators'],
            'learning_rate': best_lgb_trial.params['learning_rate'],
            'num_leaves': best_lgb_trial.params['num_leaves'],
            'min_child_samples': best_lgb_trial.params['min_child_samples'],
            'subsample': best_lgb_trial.params['subsample'],
            'r2_score': float(lgb_full_r2)
        },
        'training_info': {
            'total_samples': len(taxi_full),
            'total_features': len(features),
            'timestamp': timestamp
        }
    }
    
    with open(hyperparams_file, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    print(f"Hyperparameters saved to: {hyperparams_file}\n")
    
    # Plot feature importance comparison - ALL features

    all_features_df = feature_importance_df
    
    fig, axes = plt.subplots(1, 2, figsize=(18, max(10, len(all_features_df) * 0.25)))
    
    # XGBoost importance
    axes[0].barh(range(len(all_features_df)), all_features_df['XGBoost_Normalized'], color='#FF6B6B')
    axes[0].set_yticks(range(len(all_features_df)))
    axes[0].set_yticklabels(all_features_df['Feature'], fontsize=9)
    axes[0].set_xlabel('Normalized Importance', fontsize=11, fontweight='bold')
    axes[0].set_title(f'XGBoost Feature Importance (All {len(all_features_df)} Features)', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # LightGBM importance
    axes[1].barh(range(len(all_features_df)), all_features_df['LightGBM_Normalized'], color='#45B7D1')
    axes[1].set_yticks(range(len(all_features_df)))
    axes[1].set_yticklabels(all_features_df['Feature'], fontsize=9)
    axes[1].set_xlabel('Normalized Importance', fontsize=11, fontweight='bold')
    axes[1].set_title(f'LightGBM Feature Importance (All {len(all_features_df)} Features)', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = f'results_logs/feature_importance_comparison_ALL_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot (ALL features) saved to: {plot_file}\n")
    plt.show()
    
    # Summary statistics
    print("==============|Feature Selection Summary|==============")
    print(f"\nXGBoost Model R²: {xgb_full_r2:.6f}")
    print(f"LightGBM Model R²: {lgb_full_r2:.6f}")
    print(f"Better Model: {'XGBoost' if xgb_full_r2 > lgb_full_r2 else 'LightGBM'} ({max(xgb_full_r2, lgb_full_r2):.6f})")
    print(f"\nTotal Features Analyzed: {len(features)}")
    print(f"Top 5 Consensus Features (by average importance):")
    for i, row in feature_importance_df.head(5).iterrows():
        print(f"  {i+1}. {row['Feature']:30s} - Avg Importance: {row['Average_Importance']:.6f}")
    
    # Feature removal analysis
    print("\n==============|Feature Removal Analysis|==============")
    
    # Define importance thresholds
    thresholds = [0.005, 0.01, 0.02, 0.05]
    
    print("\nFeatures that could be removed (by importance threshold):")
    for threshold in thresholds:
        removable = feature_importance_df[feature_importance_df['Average_Importance'] <= threshold]
        print(f"\n  Below {threshold*100:.1f}% avg importance: {len(removable)} features")
        if len(removable) > 0:
            for idx, row in removable.iterrows():
                print(f"    - {row['Feature']:30s} : {row['Average_Importance']:.6f}")
    
    print(f"\nNumber of highly important features (>1% avg importance): {len(feature_importance_df[feature_importance_df['Average_Importance'] > 0.01])}")
    print(f"Number of moderately important features (0.5%-1% avg importance): {len(feature_importance_df[(feature_importance_df['Average_Importance'] > 0.005) & (feature_importance_df['Average_Importance'] <= 0.01)])}")
    print(f"Number of low importance features (<0.5% avg importance): {len(feature_importance_df[feature_importance_df['Average_Importance'] <= 0.005])}")

if ENABLE_FULL_TRAINING:
    print("\n\n==============|Full 100% Data Training with Feature Selection|==============\n")



    # Reload full dataset with all features
    df_full = pd.read_csv('Train.csv')
    taxi_full = setup_features(df_full, sample_fraction=1.0)
    x_full_raw = taxi_full[features]
    # scaler_full = StandardScaler()
    x_full = x_full_raw  # scaler_full.fit_transform(x_full_raw)
    y_full = taxi_full['duration']

    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # XGBoost — 5-fold CV for R², then full-data fit for importances
    # ------------------------------------------------------------------
    print("Training XGBoost on 100% data (5-fold CV)...")
    xgb_full = xgb.XGBRegressor(
        n_estimators=best_xgb_trial.params['n_estimators'],
        learning_rate=best_xgb_trial.params['learning_rate'],
        max_depth=best_xgb_trial.params['max_depth'],
        min_child_weight=best_xgb_trial.params['min_child_weight'],
        subsample=best_xgb_trial.params['subsample'],
        colsample_bytree=best_xgb_trial.params['colsample_bytree'],
        gamma=best_xgb_trial.params.get('gamma', 0),
        reg_alpha=best_xgb_trial.params.get('reg_alpha', 0),
        reg_lambda=best_xgb_trial.params.get('reg_lambda', 1),
        random_state=42,
        n_jobs=12,
        verbosity=0
    )

    xgb_cv_r2_scores   = cross_val_score(xgb_full, x_full, y_full, cv=cv5, scoring='r2')
    xgb_cv_rmse_scores = np.sqrt(-cross_val_score(xgb_full, x_full, y_full, cv=cv5, scoring='neg_mean_squared_error'))
    xgb_full_r2   = xgb_cv_r2_scores.mean()
    xgb_full_rmse = xgb_cv_rmse_scores.mean()

    print(f"XGBoost CV R²: {xgb_full_r2:.6f}")
    print(f"XGBoost CV RMSE: {xgb_full_rmse:.4f}")

    # Fit on full data to extract feature importances
    xgb_full.fit(x_full, y_full)
    print()

    # ------------------------------------------------------------------
    # LightGBM — 5-fold CV for R², then full-data fit for importances
    # ------------------------------------------------------------------
    print("Training LightGBM on 100% data (5-fold CV)...")
    lgb_full = lgb.LGBMRegressor(
        n_estimators=best_lgb_trial.params['n_estimators'],
        learning_rate=best_lgb_trial.params['learning_rate'],
        num_leaves=best_lgb_trial.params['num_leaves'],
        min_child_samples=best_lgb_trial.params['min_child_samples'],
        subsample=best_lgb_trial.params['subsample'],
        feature_fraction=best_lgb_trial.params.get('feature_fraction', 1.0),
        reg_alpha=best_lgb_trial.params.get('reg_alpha', 0),
        reg_lambda=best_lgb_trial.params.get('reg_lambda', 1),
        random_state=42,
        n_jobs=12,
        verbose=-1
    )

    lgb_cv_r2_scores   = cross_val_score(lgb_full, x_full, y_full, cv=cv5, scoring='r2')
    lgb_cv_rmse_scores = np.sqrt(-cross_val_score(lgb_full, x_full, y_full, cv=cv5, scoring='neg_mean_squared_error'))
    lgb_full_r2   = lgb_cv_r2_scores.mean()
    lgb_full_rmse = lgb_cv_rmse_scores.mean()

    print(f"LightGBM CV R²: {lgb_full_r2:.6f}")
    print(f"LightGBM CV RMSE: {lgb_full_rmse:.4f}")

    # Fit on full data to extract feature importances
    lgb_full.fit(x_full, y_full)
    print()

    # ------------------------------------------------------------------
    # Feature importances from the full-data fits
    # ------------------------------------------------------------------
    print("==============|Feature Importance Analysis|==============\n")

    xgb_importance = xgb_full.feature_importances_
    lgb_importance = lgb_full.feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'XGBoost_Importance': xgb_importance,
        'LightGBM_Importance': lgb_importance,
    })

    # Normalize importances to 0-1 scale for easier comparison
    xgb_norm = xgb_importance / np.sum(xgb_importance)
    lgb_norm  = lgb_importance / np.sum(lgb_importance)

    feature_importance_df['XGBoost_Normalized']  = xgb_norm
    feature_importance_df['LightGBM_Normalized'] = lgb_norm
    feature_importance_df['Average_Importance']  = (xgb_norm + lgb_norm) / 2

    feature_importance_df = feature_importance_df.sort_values('Average_Importance', ascending=False)

    print("ALL Features Ranked by Average Importance:\n")
    print(feature_importance_df[['Feature', 'XGBoost_Normalized', 'LightGBM_Normalized', 'Average_Importance']].to_string(index=False))
    print()

    # Save feature importance to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    importance_file = f'results_logs/feature_importance_full_training_{timestamp}.csv'
    feature_importance_df.to_csv(importance_file, index=False)
    print(f"\nFull feature importance saved to: {importance_file}")

    # Save model hyperparameters + CV results
    hyperparams_file = f'results_logs/final_hyperparameters_full_training_{timestamp}.json'
    hyperparams = {
        'XGBoost': {
            'n_estimators':    best_xgb_trial.params['n_estimators'],
            'learning_rate':   best_xgb_trial.params['learning_rate'],
            'max_depth':       best_xgb_trial.params['max_depth'],
            'min_child_weight':best_xgb_trial.params['min_child_weight'],
            'subsample':       best_xgb_trial.params['subsample'],
            'colsample_bytree':best_xgb_trial.params['colsample_bytree'],
            'cv_r2_mean':  float(xgb_full_r2),
            'cv_r2_std':   float(xgb_cv_r2_scores.std()),
            'cv_rmse_mean':float(xgb_full_rmse),
            'cv_rmse_std': float(xgb_cv_rmse_scores.std()),
            'cv_r2_folds': xgb_cv_r2_scores.tolist(),
            'cv_rmse_folds':xgb_cv_rmse_scores.tolist(),
        },
        'LightGBM': {
            'n_estimators':    best_lgb_trial.params['n_estimators'],
            'learning_rate':   best_lgb_trial.params['learning_rate'],
            'num_leaves':      best_lgb_trial.params['num_leaves'],
            'min_child_samples':best_lgb_trial.params['min_child_samples'],
            'subsample':       best_lgb_trial.params['subsample'],
            'cv_r2_mean':  float(lgb_full_r2),
            'cv_r2_std':   float(lgb_cv_r2_scores.std()),
            'cv_rmse_mean':float(lgb_full_rmse),
            'cv_rmse_std': float(lgb_cv_rmse_scores.std()),
            'cv_r2_folds': lgb_cv_r2_scores.tolist(),
            'cv_rmse_folds':lgb_cv_rmse_scores.tolist(),
        },
        'training_info': {
            'cv_folds':       5,
            'total_samples':  len(taxi_full),
            'total_features': len(features),
            'timestamp':      timestamp
        }
    }

    with open(hyperparams_file, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    print(f"Hyperparameters saved to: {hyperparams_file}\n")

    # Plot feature importance comparison — ALL features


    all_features_df = feature_importance_df

    fig, axes = plt.subplots(1, 2, figsize=(18, max(10, len(all_features_df) * 0.25)))

    axes[0].barh(range(len(all_features_df)), all_features_df['XGBoost_Normalized'], color='#FF6B6B')
    axes[0].set_yticks(range(len(all_features_df)))
    axes[0].set_yticklabels(all_features_df['Feature'], fontsize=9)
    axes[0].set_xlabel('Normalized Importance', fontsize=11, fontweight='bold')
    axes[0].set_title(f'XGBoost Feature Importance (All {len(all_features_df)} Features)', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    axes[1].barh(range(len(all_features_df)), all_features_df['LightGBM_Normalized'], color='#45B7D1')
    axes[1].set_yticks(range(len(all_features_df)))
    axes[1].set_yticklabels(all_features_df['Feature'], fontsize=9)
    axes[1].set_xlabel('Normalized Importance', fontsize=11, fontweight='bold')
    axes[1].set_title(f'LightGBM Feature Importance (All {len(all_features_df)} Features)', fontsize=12, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()

    plot_file = f'results_logs/feature_importance_comparison_ALL_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot (ALL features) saved to: {plot_file}\n")
    plt.show()

    # Summary statistics
    print("==============|Feature Selection Summary|==============")
    print(f"\nXGBoost CV R²: {xgb_full_r2:.6f}  |  CV RMSE: {xgb_full_rmse:.4f}")
    print(f"LightGBM CV R²: {lgb_full_r2:.6f}  |  CV RMSE: {lgb_full_rmse:.4f}")
    print(f"Better Model (CV R²): {'XGBoost' if xgb_full_r2 > lgb_full_r2 else 'LightGBM'} ({max(xgb_full_r2, lgb_full_r2):.6f})")
    print(f"\nTotal Features Analyzed: {len(features)}")
    print(f"Top 5 Consensus Features (by average importance):")
    for i, row in feature_importance_df.head(5).iterrows():
        print(f"  {i+1}. {row['Feature']:30s} - Avg Importance: {row['Average_Importance']:.6f}")

    # Feature removal analysis
    print("\n==============|Feature Removal Analysis|==============")

    thresholds = [0.005, 0.01, 0.02, 0.05]

    print("\nFeatures that could be removed (by importance threshold):")
    for threshold in thresholds:
        removable = feature_importance_df[feature_importance_df['Average_Importance'] <= threshold]
        print(f"\n  Below {threshold*100:.1f}% avg importance: {len(removable)} features")
        if len(removable) > 0:
            for idx, row in removable.iterrows():
                print(f"    - {row['Feature']:30s} : {row['Average_Importance']:.6f}")

    print(f"\nNumber of highly important features   (>1% avg importance): {len(feature_importance_df[feature_importance_df['Average_Importance'] > 0.01])}")
    print(f"Number of moderately important features (0.5%-1% avg importance): {len(feature_importance_df[(feature_importance_df['Average_Importance'] > 0.005) & (feature_importance_df['Average_Importance'] <= 0.01)])}")
    print(f"Number of low importance features       (<0.5% avg importance): {len(feature_importance_df[feature_importance_df['Average_Importance'] <= 0.005])}")


# =============================================================================
# ENABLE_FEATURE_REFINE — Re-train after dropping low-importance features
# Requires ENABLE_FULL_TRAINING to have run first (uses feature_importance_df,
# xgb_full_r2/rmse, lgb_full_r2/rmse as CV baselines).
# =============================================================================
ENABLE_FEATURE_REFINE = True   # ← toggle here

if ENABLE_FEATURE_REFINE:
    print("\n\n==============|Feature Refinement — Drop Low-Importance Features|==============\n")

    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

    # ------------------------------------------------------------------
    # 1.  Classify every feature into the three importance tiers
    # ------------------------------------------------------------------
    HIGH_THRESH = 0.01   # > 1 %      → High
    MOD_THRESH  = 0.005  # 0.5 %–1 % → Moderate
    # below MOD_THRESH   → Low

    def classify_importance(val):
        if val > HIGH_THRESH:
            return 'High'
        elif val > MOD_THRESH:
            return 'Moderate'
        else:
            return 'Low'

    feature_importance_df['XGBoost_Tier']  = feature_importance_df['XGBoost_Normalized'].apply(classify_importance)
    feature_importance_df['LightGBM_Tier'] = feature_importance_df['LightGBM_Normalized'].apply(classify_importance)
    feature_importance_df['Average_Tier']  = feature_importance_df['Average_Importance'].apply(classify_importance)

    # ------------------------------------------------------------------
    # 2.  Log each tier for XGBoost, LightGBM, and Average
    # ------------------------------------------------------------------
    for model_label, tier_col in [
        ('XGBoost',              'XGBoost_Tier'),
        ('LightGBM',             'LightGBM_Tier'),
        ('Average (Consensus)',  'Average_Tier'),
    ]:
        print(f"--- {model_label} Feature Tiers ---")
        for tier in ['High', 'Moderate', 'Low']:
            tier_features = feature_importance_df[feature_importance_df[tier_col] == tier]
            imp_col = (
                'XGBoost_Normalized'  if model_label == 'XGBoost'  else
                'LightGBM_Normalized' if model_label == 'LightGBM' else
                'Average_Importance'
            )
            print(f"\n  [{tier} Importance] — {len(tier_features)} feature(s):")
            if len(tier_features) == 0:
                print("    (none)")
            else:
                for _, row in tier_features.iterrows():
                    print(f"    {row['Feature']:35s}  {imp_col.split('_')[0]} norm: {row[imp_col]:.6f}")
        print()

    # Save the tier-annotated importance table
    timestamp_refine = datetime.now().strftime('%Y%m%d_%H%M%S')
    tier_file = f'results_logs/feature_tiers_{timestamp_refine}.csv'
    feature_importance_df.to_csv(tier_file, index=False)
    print(f"Feature tier table saved to: {tier_file}\n")

    # ------------------------------------------------------------------
    # 3.  Build the refined feature list (drop Low consensus avg tier)
    # ------------------------------------------------------------------
    low_avg_features = feature_importance_df[
        feature_importance_df['Average_Tier'] == 'Low'
    ]['Feature'].tolist()

    refined_features = [f for f in features if f not in low_avg_features]

    print(f"Features removed (Low consensus average importance): {len(low_avg_features)}")
    for f in low_avg_features:
        row = feature_importance_df[feature_importance_df['Feature'] == f].iloc[0]
        print(f"  - {f:35s}  avg: {row['Average_Importance']:.6f}  "
              f"(XGB: {row['XGBoost_Normalized']:.6f} [{row['XGBoost_Tier']}]  "
              f"LGB: {row['LightGBM_Normalized']:.6f} [{row['LightGBM_Tier']}])")

    print(f"\nOriginal feature count : {len(features)}")
    print(f"Refined  feature count : {len(refined_features)}")
    print(f"Retained features      : {refined_features}\n")

    # ------------------------------------------------------------------
    # 4.  5-fold CV on refined feature set
    # ------------------------------------------------------------------
    x_refined = x_full[refined_features]

    # --- XGBoost refined ---
    print("Running 5-fold CV for XGBoost on refined features...")
    xgb_refined = xgb.XGBRegressor(
        n_estimators=best_xgb_trial.params['n_estimators'],
        learning_rate=best_xgb_trial.params['learning_rate'],
        max_depth=best_xgb_trial.params['max_depth'],
        min_child_weight=best_xgb_trial.params['min_child_weight'],
        subsample=best_xgb_trial.params['subsample'],
        colsample_bytree=best_xgb_trial.params['colsample_bytree'],
        gamma=best_xgb_trial.params.get('gamma', 0),
        reg_alpha=best_xgb_trial.params.get('reg_alpha', 0),
        reg_lambda=best_xgb_trial.params.get('reg_lambda', 1),
        random_state=42,
        n_jobs=12,
        verbosity=0
    )

    xgb_ref_r2_scores   = cross_val_score(xgb_refined, x_refined, y_full, cv=cv5, scoring='r2')
    xgb_ref_rmse_scores = np.sqrt(-cross_val_score(xgb_refined, x_refined, y_full, cv=cv5, scoring='neg_mean_squared_error'))
    xgb_refined_r2   = xgb_ref_r2_scores.mean()
    xgb_refined_rmse = xgb_ref_rmse_scores.mean()

    print(f"XGBoost (refined) CV R²: {xgb_refined_r2:.6f}")
    print(f"XGBoost (refined) CV RMSE : {xgb_refined_rmse:.4f}")

    # --- LightGBM refined ---
    print("Running 5-fold CV for LightGBM on refined features...")
    lgb_refined = lgb.LGBMRegressor(
        n_estimators=best_lgb_trial.params['n_estimators'],
        learning_rate=best_lgb_trial.params['learning_rate'],
        num_leaves=best_lgb_trial.params['num_leaves'],
        min_child_samples=best_lgb_trial.params['min_child_samples'],
        subsample=best_lgb_trial.params['subsample'],
        feature_fraction=best_lgb_trial.params.get('feature_fraction', 1.0),
        reg_alpha=best_lgb_trial.params.get('reg_alpha', 0),
        reg_lambda=best_lgb_trial.params.get('reg_lambda', 1),
        random_state=42,
        n_jobs=12,
        verbose=-1
    )

    lgb_ref_r2_scores   = cross_val_score(lgb_refined, x_refined, y_full, cv=cv5, scoring='r2')
    lgb_ref_rmse_scores = np.sqrt(-cross_val_score(lgb_refined, x_refined, y_full, cv=cv5, scoring='neg_mean_squared_error'))
    lgb_refined_r2   = lgb_ref_r2_scores.mean()
    lgb_refined_rmse = lgb_ref_rmse_scores.mean()

    print(f"LightGBM (refined) CV R²: {lgb_refined_r2:.6f}")
    print(f"LightGBM (refined) CV RMSE: {lgb_refined_rmse:.4f}\n")

    # ------------------------------------------------------------------
    # 5.  Delta comparison: full feature set vs refined feature set
    # ------------------------------------------------------------------
    print("==============|Refined vs Full — Performance Delta|==============\n")

    xgb_r2_delta   = xgb_refined_r2   - xgb_full_r2
    xgb_rmse_delta = xgb_refined_rmse - xgb_full_rmse
    lgb_r2_delta   = lgb_refined_r2   - lgb_full_r2
    lgb_rmse_delta = lgb_refined_rmse - lgb_full_rmse

    def _delta_str(val, higher_better=True):
        sign   = '+' if val >= 0 else ''
        better = (val > 0) if higher_better else (val < 0)
        tag    = 'improved' if better else ('no change' if val == 0 else 'worse')
        return f"{sign}{val:.6f}  ({tag})"

    header = f"{'Metric':<26} {'Full Features (CV)':>22} {'Refined Features (CV)':>22} {'Delta':>30}"
    print(header)
    print('-' * len(header))
    print(f"{'XGBoost  R²':<26} {xgb_full_r2:>22.6f} {xgb_refined_r2:>22.6f} {_delta_str(xgb_r2_delta, higher_better=True):>30}")
    print(f"{'XGBoost  RMSE':<26} {xgb_full_rmse:>22.4f} {xgb_refined_rmse:>22.4f} {_delta_str(xgb_rmse_delta, higher_better=False):>30}")
    print(f"{'LightGBM R²':<26} {lgb_full_r2:>22.6f} {lgb_refined_r2:>22.6f} {_delta_str(lgb_r2_delta, higher_better=True):>30}")
    print(f"{'LightGBM RMSE':<26} {lgb_full_rmse:>22.4f} {lgb_refined_rmse:>22.4f} {_delta_str(lgb_rmse_delta, higher_better=False):>30}")
    print()

    avg_r2_full    = (xgb_full_r2    + lgb_full_r2)    / 2
    avg_r2_refined = (xgb_refined_r2 + lgb_refined_r2) / 2
    verdict = "REFINED feature set is better" if avg_r2_refined > avg_r2_full else "FULL feature set is better (or equal)"
    print(f"Average CV R² (full):    {avg_r2_full:.6f}")
    print(f"Average CV R² (refined): {avg_r2_refined:.6f}")
    print(f"Verdict: {verdict}\n")

    # ------------------------------------------------------------------
    # 6.  Plot side-by-side bar chart: full vs refined per model
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics     = ['CV R²', 'CV RMSE']
    xgb_full_v  = [xgb_full_r2,    xgb_full_rmse]
    xgb_refin_v = [xgb_refined_r2, xgb_refined_rmse]
    lgb_full_v  = [lgb_full_r2,    lgb_full_rmse]
    lgb_refin_v = [lgb_refined_r2, lgb_refined_rmse]

    x_pos = np.arange(len(metrics))
    bar_w = 0.35

    for ax, full_v, refin_v, title, color_full, color_ref in [
        (axes[0], xgb_full_v, xgb_refin_v, 'XGBoost',  '#FF6B6B', '#c0392b'),
        (axes[1], lgb_full_v, lgb_refin_v, 'LightGBM', '#45B7D1', '#1a7a9e'),
    ]:
        bars1 = ax.bar(x_pos - bar_w/2, full_v,  bar_w, label='Full features',    color=color_full, alpha=0.85)
        bars2 = ax.bar(x_pos + bar_w/2, refin_v, bar_w, label='Refined features', color=color_ref,  alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_title(f'{title}: Full vs Refined (5-Fold CV)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.01,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Full vs Refined Feature Set — 5-Fold CV Performance', fontsize=13, fontweight='bold')
    plt.tight_layout()

    refine_plot_file = f'results_logs/feature_refine_comparison_{timestamp_refine}.png'
    plt.savefig(refine_plot_file, dpi=300, bbox_inches='tight')
    print(f"Refinement comparison plot saved to: {refine_plot_file}\n")
    plt.show()

    # ------------------------------------------------------------------
    # 7.  Save refinement results to JSON
    # ------------------------------------------------------------------
    refine_results = {
        'cv_folds': 5,
        'removed_features':  low_avg_features,
        'retained_features': refined_features,
        'feature_counts': {
            'original': len(features),
            'refined':  len(refined_features),
            'removed':  len(low_avg_features)
        },
        'full_feature_cv_results': {
            'XGBoost': {
                'r2_mean':   float(xgb_full_r2),
                'r2_std':    float(xgb_cv_r2_scores.std()),
                'rmse_mean': float(xgb_full_rmse),
                'rmse_std':  float(xgb_cv_rmse_scores.std()),
                'r2_folds':  xgb_cv_r2_scores.tolist(),
                'rmse_folds':xgb_cv_rmse_scores.tolist(),
            },
            'LightGBM': {
                'r2_mean':   float(lgb_full_r2),
                'r2_std':    float(lgb_cv_r2_scores.std()),
                'rmse_mean': float(lgb_full_rmse),
                'rmse_std':  float(lgb_cv_rmse_scores.std()),
                'r2_folds':  lgb_cv_r2_scores.tolist(),
                'rmse_folds':lgb_cv_rmse_scores.tolist(),
            },
        },
        'refined_feature_cv_results': {
            'XGBoost': {
                'r2_mean':   float(xgb_refined_r2),
                'r2_std':    float(xgb_ref_r2_scores.std()),
                'rmse_mean': float(xgb_refined_rmse),
                'rmse_std':  float(xgb_ref_rmse_scores.std()),
                'r2_folds':  xgb_ref_r2_scores.tolist(),
                'rmse_folds':xgb_ref_rmse_scores.tolist(),
            },
            'LightGBM': {
                'r2_mean':   float(lgb_refined_r2),
                'r2_std':    float(lgb_ref_r2_scores.std()),
                'rmse_mean': float(lgb_refined_rmse),
                'rmse_std':  float(lgb_ref_rmse_scores.std()),
                'r2_folds':  lgb_ref_r2_scores.tolist(),
                'rmse_folds':lgb_ref_rmse_scores.tolist(),
            },
        },
        'deltas': {
            'XGBoost':  {'r2_delta': float(xgb_r2_delta),  'rmse_delta': float(xgb_rmse_delta)},
            'LightGBM': {'r2_delta': float(lgb_r2_delta),  'rmse_delta': float(lgb_rmse_delta)},
        },
        'verdict':   verdict,
        'timestamp': timestamp_refine
    }

    refine_json_file = f'results_logs/feature_refine_results_{timestamp_refine}.json'
    with open(refine_json_file, 'w') as f:
        json.dump(refine_results, f, indent=2)
    print(f"Refinement results saved to: {refine_json_file}\n")

    print("==============|Feature Refinement Complete|==============\n")