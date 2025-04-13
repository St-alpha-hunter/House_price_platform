import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib  # 用于保存模型

import os
import sys

# 设置为项目根目录（包含 data, pipline 等文件夹的目录）
project_root = os.path.abspath("..")
os.chdir(project_root)
sys.path.append(project_root)

#My_choice_features = [] ,为了提示配 features_to_use,忽略即可
def train_model(df_cleaned_features, df_cleaned, target_col="Amount_clean", model_path="models_saved/rf_model.pkl",model_cls = RandomForestRegressor ,
                features_to_use = None): ## ###默认全加是为了确保函数运行，请自己用手动添加列表

    X = df_cleaned_features[features_to_use]
    y = df_cleaned [target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    after_trained_model = model_cls(n_estimators=100, random_state=42) ###在这里更改参数
    # after_trained_model = model_cls(
    #     n_estimators=200,  # 森林中树的数量（默认100，适当增大可能提升效果）
    #     max_depth=12,  # 控制每棵树的最大深度（限制过拟合）
    #     min_samples_split=8,  # 节点分裂的最小样本数（增加可以防过拟合）
    #     max_features='sqrt',  # 每次分裂考虑的特征数（"sqrt" 在回归中表现不错）
    #     random_state=42  # 保证结果可复现
    # )
    after_trained_model.fit(X_train, y_train)
    
    # 保存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(after_trained_model, model_path)
    print("✅ 模型训练完成，保存到：", model_path)
    
    return after_trained_model, X_test, y_test
