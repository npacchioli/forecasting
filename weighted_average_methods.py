import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

def weighted_average_forecast(train, test, future_steps, lookback, weights):
    """
    Perform weighted average forecast.
    
    :param train: Training data
    :param test: Test data
    :param future_steps: Number of future steps to forecast
    :param lookback: Number of periods to look back (3, 6, or 12)
    :param weights: List of weights for each period, should sum to 1
    :return: Forecast values
    """
    if len(weights) != lookback:
        raise ValueError("Number of weights must match the lookback period")
    
    if abs(sum(weights) - 1) > 1e-6:
        raise ValueError("Weights must sum to 1")
    
    # Reverse weights so that the most recent period has the last weight
    weights = weights[::-1]
    
    # Calculate weighted average
    weighted_avg = np.average(train[-lookback:], weights=weights)
    
    # Create forecast
    forecast = np.full(len(test) + future_steps, weighted_avg)
    
    return forecast

def optimize_weighted_average(train, test):
    """
    Find the best lookback period and weights for the weighted average model.
    
    :param train: Training data
    :param test: Test data
    :return: Best parameters and RMSE
    """
    lookback_options = [3, 6, 12]
    best_rmse = float('inf')
    best_params = None
    
    for lookback in lookback_options:
        # Generate some weight combinations
        weight_options = [
            [1/lookback] * lookback,  # Equal weights
            np.linspace(0.5/lookback, 1.5/lookback, lookback),  # Linearly increasing weights
            np.exp(np.linspace(np.log(0.5/lookback), np.log(1.5/lookback), lookback))  # Exponentially increasing weights
        ]
        
        for weights in weight_options:
            # Normalize weights to sum to 1
            weights = np.array(weights) / sum(weights)
            
            forecast = weighted_average_forecast(train, test, 0, lookback, weights.tolist())
            rmse = sqrt(mean_squared_error(test, forecast[:len(test)]))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'lookback': lookback, 'weights': weights.tolist()}
    
    return best_params, best_rmse