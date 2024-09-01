import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
np.float_ = np.float64
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from tabulate import tabulate

from optimizations import optimize_arima, optimize_sarima, optimize_prophet
from weighted_average_methods import optimize_weighted_average, weighted_average_forecast

warnings.filterwarnings('ignore')

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

def prepare_data(group):
    group_data = data[data['Group'] == group].set_index('Date')['Sales']
    return group_data

groups = data['Group'].unique()

print(data.head(30))


def arima_forecast(train, test, future_steps, order=(1,1,1)):
    model = ARIMA(train, order = order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test) + future_steps)
    return forecast

def sarima_forecast(train, test, future_steps, order, seasonal_order):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test) + future_steps)
    return forecast

def prophet_forecast(train, test, future_steps, params):
    # Prepare data for Prophet
    train_prophet = pd.DataFrame({'ds': train.index, 'y': train.values})

    model = Prophet(**params)
    model.fit(train_prophet)

    future = pd.DataFrame({'ds': pd.date_range(start=train.index[-1], periods=len(test) + future_steps, freq='M')})
    forecast = model.predict(future)

    return forecast['yhat']
    

def evaluate_forecast(actual, forecast):
    rmse = sqrt(mean_squared_error(actual, forecast))
    return rmse

def forecast_group(group, method='arima'):
    group_data = prepare_data(group)
    train = group_data[:-3]  # Use all but last 3 months for training
    test = group_data[-3:]   # Use last 3 months for testing
    
    future_steps = 6
    
    if method == 'arima':
        best_order = optimize_arima(train, test)
        print(f'Best ARIMA order for {group}: {best_order}')
        forecast = arima_forecast(train, test, future_steps, best_order)
        best_params = best_order
    elif method == 'sarima':
        best_order, best_seasonal_order = optimize_sarima(train, test)
        print(f'Best SARIMA order for {group}: {best_order}, seasonal_order: {best_seasonal_order}')
        forecast = sarima_forecast(train, test, future_steps, best_order, best_seasonal_order)
        best_params = (best_order, best_seasonal_order)
    elif method == 'prophet':
        best_params, _ = optimize_prophet(train, test)
        print(f'Best Prophet parameters for {group}: {best_params}')
        forecast = prophet_forecast(train, test, future_steps, best_params)
    elif method == 'weighted_average':
        best_params, _ = optimize_weighted_average(train, test)
        print(f'Best Weighted Average parameters for {group}: {best_params}')
        forecast = weighted_average_forecast(train, test, future_steps, best_params['lookback'], best_params['weights'])
        
    test_forecast = forecast[:len(test)]
    future_forecast = forecast[len(test):]
        
    rmse = evaluate_forecast(test, test_forecast)
    return test_forecast, future_forecast, rmse, train, test, best_params

def plot_forecast(group, train, test, test_forecast, future_forecast, best_params, method):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train.values, label='Historical Data')
    plt.plot(test.index, test.values, label='Actual Test Data')
    plt.plot(test.index, test_forecast, label='Test Forecast', color='red')
    
    last_date = test.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_forecast), freq='M')
    plt.plot(future_dates, future_forecast, label='Future Forecast', color='green', linestyle='--')
    
    plt.title(f'Sales Forecast for {group} - {method.upper()}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Initialize a list to store comparison results
comparison_results = []

for group in groups:
    print(f'\nForecasting for {group}')
    
    group_results = {'Group': group}
    
    for method in ['arima', 'sarima', 'prophet', 'weighted_average']:
        test_forecast, future_forecast, rmse, train, test, best_params = forecast_group(group, method)
        print(f'{method.upper()} RMSE: {rmse:.2f}')
        print(f'Forecast for test period: {test_forecast}')
        print(f'Forecast for next 6 months: {future_forecast}')
        
        # Store RMSE in the results dictionary
        group_results[f'{method.upper()} RMSE'] = round(rmse, 2)
        
        # Plot the forecast
        plot_forecast(group, train, test, test_forecast, future_forecast, best_params, method)
    
    comparison_results.append(group_results)

# Create a DataFrame from the comparison results
comparison_df = pd.DataFrame(comparison_results)

# Display the comparison table
print("\nComparison of ARIMA, SARIMA, Prophet, and Weighted Average RMSE values:")
print(comparison_df.to_string(index=False))