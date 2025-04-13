import os
import sys
import pandas as pd
from features_wlh.features_wlh import add_selected_features
from features_wlh.feature_analysis import plot_feature_correlation
from features_wlh.feature_vif_validation import check_multicollinearity
from utils.path_helper import get_data_path
from pipline.pipline import pipeline_house_data
from model.evaluate import evaluate_model
from model.train_model import train_model
from utils.config import KEYWORDS
import numpy as np

#路径管理
#设置为项目根目录（包含 data, pipline 等文件夹的目录）
project_root = os.path.abspath("..")
os.chdir(project_root)
sys.path.append(project_root)

if __name__ == "__main__":
    print("starting....")


#选择特征值,自己去填写
My_features =  ["Bathroom","Furnishing_giving","floor_level_normalize","has_amenities",
                                    "has_green_space","has_proximity","is_affordable","is_basement","is_deal",
                                    "is_gated","is_ground","is_luxury","is_marketing_strong","is_new","is_prime_location",
                                    "is_resale","is_spacious","is_well_planned","location_rank","max_floor","normal_Carpet_Area",
                                    "normal_price","ownership_score","floor_area_combo","location_comfort_combo", #有点用但不多
                                    "location_ownership_combo", "facing_height_combo", "area_furnishing_combo"]
print("✅step1--录入特征值完成")

#读取数据
df = pd.read_csv(get_data_path("house_prices.csv"))
print("✅step2--读取数据完成")

#清洗数据 or 加载已清洗数据
processed_path = get_data_path("processed_data/house_prices_cleaned.csv")
force_clean = False  # 设置为 True 可以强制重新清洗

if os.path.exists(processed_path) and not force_clean:
    df_cleaned = pd.read_csv(processed_path)
    print("✅step3--加载已清洗数据完成, shape:", df_cleaned.shape)
else:
    df_cleaned = pipeline_house_data(df, keywords=KEYWORDS)
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_cleaned.to_csv(processed_path, index=False)
    print("✅step3--首次清洗数据完成并保存, shape:", df_cleaned.shape)

#构建特征工程 填写features_to_use
df_cleaned_features = add_selected_features(df_cleaned, features_to_use = My_features) 
print("✅step4--特征工程完成")

#检验特征相关性
plot_feature_correlation(df_cleaned_features, features = My_features)
vif_result = check_multicollinearity(df_cleaned_features,features=My_features, threshold=5)
print("✅step5--特征检验完成")

#训练模型   ##模型参数进入train_model.py去调  填写features_to_use
after_trained_model, X_test, y_test = train_model(df_cleaned_features, df_cleaned, features_to_use = My_features)
print("✅step6--完成训练")

#评估模型
evaluate_model(after_trained_model, X_test, y_test)
print("✅step7--完成评估")
