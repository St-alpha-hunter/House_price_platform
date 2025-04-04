import os
import pandas as pd
from utils.config import KEYWORDS
from features.feature_pipeline import FeatureEngineer
from model.train_model import train_model
from model.evaluate import evaluate_model

# Step 1: 读取数据（优先读取清洗后的）
processed_path = "data/processed_data/house_prices_cleaned.csv"

if os.path.exists(processed_path):
    df_cleaned = pd.read_csv(processed_path)
    print("已加载清洗后数据:", df_cleaned.shape)
else:
    df_raw = pd.read_csv("data/raw_data/house_prices.csv")
    fe = FeatureEngineer(keywords=KEYWORDS, balcony_strategy="leave")
    df_cleaned = fe.transform(df_raw)
    os.makedirs("data/processed_data", exist_ok=True)
    df_cleaned.to_csv(processed_path, index=False)
    print("数据清洗完成并保存，shape:", df_cleaned.shape)

# Step 2: 训练与评估模型（可替换模型类型）
model_type = "linear"  # 可选："linear", "decision_tree", "hist_gbdt"
model_path = f"results/models/{model_type}_model.pkl"
os.makedirs("results/models", exist_ok=True)

model, X_test, y_test = train_model(df_cleaned, model_type=model_type, model_path=model_path)
evaluate_model(model, X_test, y_test)