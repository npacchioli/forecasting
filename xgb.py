import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from math import sqrt
from scipy.stats import uniform, randint


def prepare_data_for_xgboost_multi_group(data, lag=3):
    # Convert Date to datetime if it's not already
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Sort the data by Group and Date
    data = data.sort_values(['Group', 'Date'])
    
    # Create month and year features
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    
    # Initialize columns for lag and rolling statistics
    for i in range(1, lag+1):
        data[f'lag_{i}'] = np.nan
    data['rolling_mean'] = np.nan
    data['rolling_std'] = np.nan
    
    # Process each group separately
    for group in data['Group'].unique():
        group_data = data[data['Group'] == group].copy()
        
        # Create lag features
        for i in range(1, lag+1):
            group_data[f'lag_{i}'] = group_data['Sales'].shift(i)
        
        # Create rolling statistics
        group_data['rolling_mean'] = group_data['Sales'].rolling(window=3).mean()
        group_data['rolling_std'] = group_data['Sales'].rolling(window=3).std()
        
        # Update the main dataframe
        data.loc[data['Group'] == group] = group_data
    
    # Drop rows with NaN values
    data = data.dropna()
    
    # Reset index
    data = data.reset_index(drop=True)
    
    return data

def prepare_data(group):
    group_data = data[data['Group'] == group].set_index('Date')['Sales']
    return group_data

def prepare_data(df, group='Viennoiseries'):
    group_data = df[df['Group'] == group].copy()
    X = group_data.drop(['Group','Date','Sales'], axis = 1)
    y = group_data['Sales']
    return X, y

def train_predict_xgboost(X_train, y_train, X_test, params):
    model = XGBRegressor(**params, random_state =42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

def optimize_xgboost(X, y, param_distributions, n_iter = 100):
    tscv = TimeSeriesSplit(n_splits = 5)
    
    def objective(params):
        model = XGBRegressor(**params, random_state = 42)
        scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            scores.append(mse)
        return np.mean(scores)
    
    best_params = None
    best_score = float('inf')
    
    for _ in range(n_iter):
        params = {k: v.rvs() for k, v in param_distributions.items()}
        score = objective(params)
        if score < best_score:
            best_score = score
            best_params = params
            
    return best_params, best_score



data = pd.read_csv('group_pivot.csv')

if 'Total' in data.columns:
    data = data.drop('Total', axis=1)
    print('removing Total column')

# Convert column names to datetime, excluding the 'Group' column
date_columns = [pd.to_datetime(col) for col in data.columns if col != 'Group']
data.columns = ['Group'] + date_columns

# Melt the dataframe
data = data.melt(id_vars=['Group'], var_name='Date', value_name='Sales')
data = data.sort_values(['Group', 'Date'])

groups = data['Group'].unique()

xgb_data = prepare_data_for_xgboost_multi_group(data, 3)
print(xgb_data)


# 1. Prepare data for Viennoiseries
X, y = prepare_data(xgb_data, 'Pastries')

# 2. Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. Define parameter distributions for optimization
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

# 4. Optimize hyperparameters
best_params, best_score = optimize_xgboost(X_train, y_train, param_distributions)
print("Best parameters:", best_params)
print("Best MSE score:", best_score)

# 5. Train final model with best parameters and make predictions
final_predictions = train_predict_xgboost(X_train, y_train, X_test, best_params)

# 6. Evaluate final model
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Final RMSE:", final_rmse)

# 7. Plot actual vs predicted values
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, final_predictions, label='Predicted')
plt.title('Bread Sales: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()