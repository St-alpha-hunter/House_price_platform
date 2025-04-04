import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import joblib  # 用于保存模型

def train_model(df, target_col="Amount_clean", model_type="random_forest", model_path=None):
    if model_path is None:
        model_path = f"results/models/{model_type}_model.pkl"
    # 选择所有数值型特征
    features = df.select_dtypes(include=[float, int]).columns.tolist()
    features = [col for col in df.columns if col not in [target_col, "Index"]]

    # 特征和标签
    X = df[features]
    y = df[target_col]

    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 根据选择使用不同模型
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "hist_gbdt":
        model = HistGradientBoostingRegressor(random_state=42)
    elif model_type == "linear":
        model = LinearRegression()
    elif model_type == "decision_tree":
        model = DecisionTreeRegressor(random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 模型训练
    model.fit(X_train, y_train)
    
    # 保存模型
    joblib.dump(model, model_path)
    print("模型训练完成，保存到：", model_path)
    
    return model, X_test, y_test
