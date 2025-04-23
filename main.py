import os
import sys
import joblib
import pandas as pd
import numpy as np
import tabulate as tb

from utils.config import KEYWORDS
from utils.path_helper import get_data_path
from pipline.pipline import pipeline_house_data

from features_wlh.features_wlh import add_selected_features
from features_wlh.feature_analysis import plot_feature_correlation
from features_wlh.feature_vif_validation import check_multicollinearity
from features_wlh.feature_selector import feature_selection_by_k,select_final_top_features
from features_wlh.FeatureDeepAnalysis import FeatureDeepAnalysis

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from model.train_model import train_model
from model.evaluate import evaluate_model
from advanced_modify.cross_validation import enhanced_cross_validate




#路径管理
#设置为项目根目录（包含 data, pipline 等文件夹的目录）
project_root = os.path.abspath("..")
os.chdir(project_root)
sys.path.append(project_root)

if __name__ == "__main__":
    print("starting....")


#选择特征值,自己去填写
My_features =  [
    "Car_Parking",
    "Bathroom",
    "Furnishing_giving",
    "Transaction_giving",
    "balcony_rank",
    "is_ground",
    "quality_score",
    "location_rank",
    "floor_level_normalize",
    "ownership_score",
    "relative_height",
    "society_level_hot", 
    "std_Carpet_Area",
    "facing_giving",
    "Status_giving",
    "ownership_score",
    "is_multi_bathroom",
    "is_popular_location",
    "floor_area_combo",
    "location_comfort_combo",
    "floor_facing_score",
    "location_ownership_combo",
    "facing_height_combo",
    "area_furnishing_combo"
]
print("✅step1--录入我选的完成")

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
print("✅step4--特征值导入完成")

# 分析每个特征数下的最优组合
df_result = feature_selection_by_k(df_cleaned[My_features], target_col = df_cleaned["Amount_clean"], max_k=20, rank_features=10, model_cls = RandomForestRegressor)
# 直接选出最终最重要的15个特征
top_features = select_final_top_features(df_cleaned[My_features], target_col = df_cleaned["Amount_clean"], max_k=20, top_k=15, model_cls = RandomForestRegressor)
print("✅step4(1)---特征值筛选完成")

#特征向量相关性 + 自动剔除高相关
advanced_feature = plot_feature_correlation(df_cleaned_features, features = top_features, threshold=0.8)
print("✅step4(2)---剔除高相关性特征值")

#VIF检验 + 自动剔除高度VIF
advanced_features_ultimate = check_multicollinearity(df_cleaned_features, features=advanced_feature,threshold=5)
print("✅step4(3)---VIF检验")

#特征值重要性排序
deep_analysis = FeatureDeepAnalysis(df_cleaned,features=advanced_features_ultimate,model_cls=RandomForestRegressor,target_col="Amount_clean")
deep_analysis.plot_feature_importance()
print("✅step4(4)---特征值重要性排序")

#特征值KDE画图
deep_analysis.plot_feature_distribution()
print("✅step4(5)---特征值画图")

#特征值和目标变量的相关性
deep_analysis.plot_feature_vs_target()
print("✅step4(6)---特征值和目标变量的相关性")
print("特征工程结束，开始训练模型")

#训练模型   ##模型参数进入train_model.py去调  填写features_to_use
after_trained_model, X_test, y_test, X_train, y_trai = train_model(df_cleaned_features, df_cleaned, features_to_use = My_features)
print("✅step5--完成训练")

#评估模型
evaluate_model(after_trained_model, X_test, y_test)
#交叉检验
df_cross_evaulation = enhanced_cross_validate(model = RandomForestRegressor(),
                                               features = df_cleaned[advanced_features_ultimate], 
                                               target_col=df_cleaned["Amount_clean"].values, 
                                               return_df=True)
print("交叉检验结果如下:",df_cross_evaulation)
print("✅step6--完成评估")


# 保存模型
project_root = os.path.abspath(os.path.dirname(__file__))  # 当前文件所在目录
model_dir = os.path.join(project_root, "models_saved")
os.makedirs(model_dir, exist_ok=True)
model_save_path = os.path.join(model_dir, "my_model.pkl")
joblib.dump(after_trained_model, model_save_path)
print("✅step8-模型已保存")




# ==== 自动生成报告 ====


from datetime import datetime

# ==== 收集信息 ====
report_lines = []

# 报告头部
report_lines.append("# 模型训练报告")
report_lines.append(f" 日期：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("---\n")

# 使用的特征列表
report_lines.append(" 最终使用的特征（通过多轮筛选 + 去除共线性）")
for i, feat in enumerate(advanced_features_ultimate, 1):
    report_lines.append(f"{i}. {feat}")
report_lines.append("")

# 模型评估
score = after_trained_model.score(X_test, y_test)
report_lines.append("## 模型评估结果")
report_lines.append(f"- R² 分数（score）: `{score:.4f}`")
report_lines.append(f"- 交叉检验结果:")
from tabulate import tabulate
markdown_table = tabulate(df_cross_evaulation, headers='keys', tablefmt='github', showindex=False)
report_lines.append(markdown_table)

# 你可以加 RMSE、MAE，比如：
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# rmse = mean_squared_error(y_test, after_trained_model.predict(X_test), squared=False)
# report_lines.append(f"- RMSE: `{rmse:.2f}`")
report_lines.append("")

# 模型参数
report_lines.append("## 模型类型")
report_lines.append(f"- 使用模型：`{type(after_trained_model).__name__}`")

# 输出保存信息
report_lines.append("\n## 模型已保存路径")
report_lines.append("- `models_saved/my_model.pkl`")

report_lines.append("\n---")
report_lines.append("*报告由自动脚本生成。*")

# ==== 写入文件 ====
with open("model_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))


#保存文件
project_root = os.path.abspath(os.path.dirname(__file__))  # 当前文件所在目录
report_dir = os.path.join(project_root, "report")
os.makedirs(report_dir, exist_ok=True)
report_save_path = os.path.join(report_dir, "report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

print("✅ step9 - 训练报告已自动生成", report_save_path)
