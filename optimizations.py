import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
np.float_ = np.float64
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

def optimize_arima(train, test):
    best_rmse = float('inf')
    best_order = None
    p_values = range(0,3)
    d_values = range(0,2)
    q_values = range(0,3)
    
    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(train, order=(p,d,q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test))
            rmse = sqrt(mean_squared_error(test, forecast))
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = (p, d, q)
        except:
            continue
        
        return best_order
    
    
def optimize_sarima(train, test):
    best_rmse = float('inf')
    best_order = None
    best_seasonal_order = None
    p_values = range(0,3)
    d_values = range(0,2)
    q_values = range(0,3)
    P_values = range(0,2)
    D_values = range(0,2)
    Q_values = range(0,2)
    m_values = [12] #monthly forecast
    
    for p,d, q, P, D, Q, m in itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, m_values):
        try:
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
            model_fit = model.fit(disp=False)
            forecast = model_fit.forecast(steps=len(test))
            rmse = sqrt(mean_squared_error(test, forecast))
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = (p, d, q)
                best_seasonal_order = (P, D, Q, m)
        except:
            continue
        
    return best_order, best_seasonal_order

def optimize_prophet(train, test):
    best_params = None
    best_rmse = float('inf')

    # Define parameter grid
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    # Prepare data for Prophet
    train_prophet = pd.DataFrame({'ds': train.index, 'y': train.values})

    for changepoint_prior_scale in param_grid['changepoint_prior_scale']:
        for seasonality_prior_scale in param_grid['seasonality_prior_scale']:
            for holidays_prior_scale in param_grid['holidays_prior_scale']:
                for seasonality_mode in param_grid['seasonality_mode']:
                    model = Prophet(
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale,
                        holidays_prior_scale=holidays_prior_scale,
                        seasonality_mode=seasonality_mode
                    )
                    model.fit(train_prophet)

                    future = pd.DataFrame({'ds': test.index})
                    forecast = model.predict(future)
                    rmse = sqrt(mean_squared_error(test.values, forecast['yhat']))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = {
                            'changepoint_prior_scale': changepoint_prior_scale,
                            'seasonality_prior_scale': seasonality_prior_scale,
                            'holidays_prior_scale': holidays_prior_scale,
                            'seasonality_mode': seasonality_mode
                        }

    return best_params, best_rmse