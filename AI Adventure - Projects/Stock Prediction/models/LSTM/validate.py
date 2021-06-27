# test accuracy of model
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mse = mean_squared_error(predictions,y_test_unscaled)
rmse = math.sqrt(mse)
mae = mean_absolute_error(predictions, y_test_unscaled)
mape = mean_absolute_percentage_error(predictions, y_test_unscaled)
print(mse)
print(rmse)
print(mae)
print(mape)
