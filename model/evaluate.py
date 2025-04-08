from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model.train_model import train_model
import numpy as np


def evaluate_model(after_trained_model, X_test, y_test):
    
    y_pred = after_trained_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
