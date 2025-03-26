import numpy as np
#from model import predict_rating 

def test_rating_accuracy(predictions, ground_truth):

    # Calculate MAE and RMSE
    mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truth)))
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(ground_truth))**2))
    within_tolerance = np.mean(np.abs(np.array(predictions) - np.array(ground_truth)) <= 0.5) # Calculate % of predictions within ±0.5

    return mae, rmse, within_tolerance