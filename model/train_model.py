import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib  # 用于保存模型

#My_choice_features = [] ,为了提示配 features_to_use,忽略即可
def train_model(df_cleaned_features, df_cleaned, target_col="Amount_clean", model_path="rf_model.pkl",model_cls = RandomForestRegressor ,
                features_to_use = None): ## ###默认全加是为了确保函数运行，请自己用手动添加列表

    X = df_cleaned_features[features_to_use]
    y = df_cleaned [target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    after_trained_model = model_cls(n_estimators=100, random_state=42) ###在这里更改参数
    after_trained_model.fit(X_train, y_train)
    
    # 保存模型
    joblib.dump(after_trained_model, model_path) ##根据模型真实名称命名
    print("✅ 模型训练完成，保存到：", model_path)
    
    return after_trained_model, X_test, y_test
