import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib  # 用于保存模型

def train_model(df, target_col="Amount(in rupees)", model_path="rf_model.pkl"):
    # 特征选择（你可以修改）
    features = [col for col in df.columns if col not in [target_col, "Index"]]
    
    X = df[features]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 保存模型
    joblib.dump(model, model_path)
    print("✅ 模型训练完成，保存到：", model_path)
    
    return model, X_test, y_test
