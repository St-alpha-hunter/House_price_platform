import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from utils.config import KEYWORDS
import random
from pipline.pipline import pipeline_house_data


# 1. 特征函数注册字典
FEATURE_FUNCTIONS = {}

def register_feature(name):
    def decorator(func):
        FEATURE_FUNCTIONS[name] = func
        return func
    return decorator

#f0   原生price含缺失值
@register_feature("price") 
def price(df):
    df["price"] = df["Price (in rupees)"]
    return df

#f1  #填充price
@register_feature("std_price")
def std_price(df):
    df_train = df[df["Price (in rupees)"].notna()]
    df_missing = df[df["Price (in rupees)"].isna()]
    # 可选特征（你可以换掉）
    features = [
        "Carpet Area", "floor_level", "max_floor", "location_encoded",
        "is_affordable", "is_luxury", "has_amenities"
    ]
    X = df_train[features]
    y = df_train["Price (in rupees)"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df.loc[df["Price (in rupees)"].isna(), "Price (in rupees)"] = model.predict(df_missing[features])

    df["std_price"] = (df["Price (in rupees)"] - df["Price (in rupees)"].mean())/df["Price (in rupees)"].std()
    return df

#f2
@register_feature("normal_price")
def normal_price(df):
    df_train = df[df["Price (in rupees)"].notna()]
    df_missing = df[df["Price (in rupees)"].isna()]

    features = [
        "Carpet Area", "floor_level", "max_floor", "location_encoded",
        "is_affordable", "is_luxury", "has_amenities"
    ]

    if not df_missing.empty and not df_missing[features].empty:
        X = df_train[features]
        y = df_train["Price (in rupees)"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        df.loc[df["Price (in rupees)"].isna(), "Price (in rupees)"] = model.predict(df_missing[features])

    # 正常归一化
    df["normal_price"] = (df["Price (in rupees)"] - df["Price (in rupees)"].mean()) / (
        df["Price (in rupees)"].max() - df["Price (in rupees)"].min()
    )

    return df
#f3
@register_feature("std_Carpet_Area")
def std_Carpet_Area(df):
    df["std_Carpet_Area"] = (df["Carpet Area"] - df["Carpet Area"].mean())/df["Carpet Area"].std()
    return df

#f4
@register_feature("normal_Carpet_Area")
def normal_price(df):
    df["normal_Carpet_Area"] = (df["Carpet Area"] - df["Carpet Area"].mean())/(df["Carpet Area"].max() - df["Carpet Area"].min())
    return df

#f5
@register_feature("Transaction_giving")
def Transaction_giving(df):
    map_transaction = {'New Property':2, 'Rent/Lease':1, "Other":0, "Resale":-1}
    df["Transaction_giving"] = df["Transaction"].map(map_transaction)
    return df

#f6
@register_feature("Trasaction_hot")
def Transaction_hot(df):
    dummies = pd.get_dummies(df["Transaction"], prefix="Transaction")
    df = pd.concat([df, dummies], axis=1)  #列 横向拼接（axis=1）
    return df

#f7
@register_feature("Furnishing_hot")
def Furnishing_hot(df):
    dummies = pd.get_dummies(df["Furnishing"], prefix="Furnishing")
    df = pd.concat([df, dummies], axis=1)  #列 横向拼接（axis=1）
    return df  

#f8
@register_feature("Furnishing_giving")
def Furnishing_giving(df):
    map_furnishing = {'Unfurnished':0, 'Semi-Furnished':1,'Furnished':2}
    df["Furnishing_giving"] = df["Furnishing"].map(map_furnishing)
    return df  

#f9
@register_feature("Status_giving")
def Status_giving(df):
    map_status = {'Ready to Move':1, 'Not allowed to Move':0}
    df["Status_giving"] = df["Status"].map(map_status)
    return df

#10
@register_feature("Status_hot")
def Status_hot(df):
    dummies = pd.get_dummies(df["Status"], prefix="Status")
    df = pd.concat([df, dummies], axis=1)  #列 横向拼接（axis=1）
    return df  

#11
@register_feature("facing_giving")
def facing_giving(df):
    df["facing_giving"] = df["col_facing_score"]
    return df

#12
@register_feature("facing_hot")
def facing_hot(df):
    dummies = pd.get_dummies(df["col_facing_score"], prefix="col_facing_score")
    df = pd.concat([df, dummies], axis=1)
    return df


#13
@register_feature("Bathroom")
def Bathroom(df):
    df["Bathroom"] = df["col_Bathroom_score"]
    return df

#14
@register_feature("balcony_rank")
def balcony_rank(df):
    map_balcony = {'1':0, '2':0, '3':1 ,'4':1, '5':1, '6':2, '7':2, '8':2, '9':2, '10':2, '>10':2}
    df["balcony_rank"] = df["Balcony"].map(map_balcony)
        # 2. 拆分训练数据 & 待预测数据
    df_known = df[df["balcony_rank"].notna()]
    df_missing = df[df["balcony_rank"].isna()]
    
    if not df_missing.empty:
        # 3. 用你已有的变量作为特征
        features = [
            "Carpet Area", "floor_level", "max_floor", "location_encoded",
            "is_affordable", "is_luxury", "has_green_space"
        ]

        # 防止缺列或者部分缺失
        df_train = df_known.dropna(subset=features)
        X = df_train[features]
        y = df_train["balcony_rank"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 预测缺失
        pred = model.predict(df_missing[features]).round().astype(int).clip(0, 2)

        # 填回原表
        df.loc[df["balcony_rank"].isna(), "balcony_rank"] = pred

    df["balcony_rank"] = df["balcony_rank"].astype(int)
    return df

#15
@register_feature("Car_Parking")
def Car_Parking(df):
    df["Car_Parking"] = df ["col_car_parking_score"]
    return df

#16
@register_feature("is_prime_location")
def is_prime_location(df):
    df["is_prime_location"] = df["is_prime_location"]
    return df

#17
@register_feature("has_proximity")
def has_proximity(df):
    df["has_proximity"] = df["has_proximity"]
    return df

#18
@register_feature("is_well_planned")
def is_well_planned(df):
     df["is_well_planned"] = df["is_well_planned"]
     return df

#19
@register_feature("is_new")
def is_new(df):
    df["is_new"] = df["is_new"]
    return df

#20
@register_feature("is_deal")
def is_deal(df):
    df["is_deal"] = df["is_deal"]
    return df

#21
@register_feature("is_gated")
def is_gated(df):
    df["is_gated"] = df["is_gated"]
    return df

#22
@register_feature("has_amenities")
def has_amenities(df):
    df["has_amenities"] = df["has_amenities"]
    return df

#23
@register_feature("has_green_space")
def has_green_space(df):
    df["has_green_space"] = df["has_green_space"]
    return df

#24
@register_feature("is_marketing_strong")
def is_marketing_strong(df):
    df["is_marketing_strong"] = df["is_marketing_strong"]
    return df

#25
@register_feature("is_resale")
def is_new(df):
    df["is_resale"] = df["is_resale"]
    return df

#26
@register_feature("is_affordable")
def is_affordable(df):
    df["is_affordable"] = df["is_affordable"]
    return df

#27
@register_feature("is_spacious")
def is_spacious(df):
    df["is_spacious"] = df["is_spacious"]
    return df

#28
@register_feature("is_luxury")
def is_luxury(df):
    df["is_luxury"] = df["is_luxury"]
    return df

#29
@register_feature("society_level_hot")
def society_level_hot(df):
    map_hot = {0:4, 1:3, 2:2, 3:1, 4:0}
    df["society_level_hot"] = df["society_level"].map(map_hot)
    return df


#30
@register_feature("location_log_encode")
def location_log_encode(df):
    df["location_freq_log"] = np.log1p(df["location_encoded"])
    df["location_log_encode"] = (df["location_freq_log"] - df["location_freq_log"].min()) / \
        (df["location_freq_log"].max() - df["location_freq_log"].min())
    df = df.drop(columns = ["location_freq_log"])
    return df

#31
@register_feature("location_rank")
def location_rank(df):
    df["location_rank"] = df["location_encoded"].rank(method="min", ascending=True)
    df["location_level"] = pd.qcut(df["location_rank"], q=5, labels=False)
    df = df.drop(columns = ["location_rank"])
    return df


#33
@register_feature("is_ground")
def is_ground(df):
    df["is_ground"] = df["is_ground"]
    return df

#34
@register_feature("is_basement")
def is_basement(df):
    df["is_basement"] = df["is_basement"]
    return df

#35
@register_feature("relative_height")
def relative_height(df):
    df["relative_height"] = df["relative_height"]
    return df

#34 ##写错了没有这个
#@register_feature("floor")
#   def floor(df):
#        df["floor"] = df["floor"]
#           return df

#35
@register_feature("floor_level_standard")
def flooe_level_standard(df):
    mean = df["max_floor"].mean()
    std = df["max_floor"].std()
    df["max_level_standard"] = (df["max_floor"] - mean) / (std + 1e-8)
    return df
#36
@register_feature("floor_level_normalize")
def floor_level_normalize(df):
    min_val = df["max_floor"].min()
    max_val = df["max_floor"].max()
    df["floor_level_normalize"] = (df["max_floor"] - min_val) / (max_val - min_val + 1e-8) ##防止除以 0 或极小的数，以避免数值计算出错（比如 NaN、inf）
    return df

#36
@register_feature("max_floor")
def max_floor(df):
    df["max_floor"] = df["max_floor"]
    return df

#37
@register_feature("max_floor_standard")
def max_floor_standard(df):
    mean = df["max_floor"].mean()
    std = df["max_floor"].std()
    df["max_floor_standard"] = (df["max_floor"] - mean) / (std + 1e-8)
    return df

#38
@register_feature("max_floor_normalize")
def max_floor_normalize(df):
    min_val = df["max_floor"].min()
    max_val = df["max_floor"].max()
    df["max_floor_normalize"] = (df["max_floor"] - min_val) / (max_val - min_val + 1e-8)
    return df

#39
@register_feature("ownership_score")
def ownership_score(df):
    df["ownership_score"] = df ["ownership_score"]
    return df

# 41 是否低于本地中位价
@register_feature("is_undervalued")
def is_undervalued(df):
    df["median_price_per_location"] = df.groupby("location_encoded")["Amount_clean"].transform("median")
    df["is_undervalued"] = (df["Amount_clean"] < df["median_price_per_location"]).astype(int)
    df = df.drop(columns=["median_price_per_location"])
    return df

# 42 房源是否属于“大户型”
@register_feature("is_large_house")
def is_large_house(df):
    df["is_large_house"] = (df["Carpet Area"] > 1500).astype(int)
    return df

# 43 房子是否有多个卫生间
@register_feature("is_multi_bathroom")
def is_multi_bathroom(df):
    df = FEATURE_FUNCTIONS["Bathroom"](df)
    df["is_multi_bathroom"] = (df["Bathroom"] >= 2).astype(int)
    return df

# 44 是否位于热门地段
@register_feature("is_popular_location")
def is_popular_location(df):
    median_freq = df["location_encoded"].median()
    df["is_popular_location"] = (df["location_encoded"] >= median_freq).astype(int)
    return df

# 46 楼层高度与房屋面积交叉,捕捉“高层 + 大面积”倾向于更高价值的潜在规律
@register_feature("floor_area_combo")
def floor_area_combo(df):
    if "floor_level" in df.columns and "Carpet Area" in df.columns:
        df["floor_area_combo"] = df["floor_level"] * df["Carpet Area"]
    else:
        df["floor_area_combo"] = 0  # 或者 raise error
    return df

# 47 高地段 + 绿化 + 配套完善，意味着 稀缺性 +居住价值更高，特别适合家庭购房需求
@register_feature("location_comfort_combo")
def location_comfort_combo(df):
    if all(col in df.columns for col in ["location_level", "has_amenities", "has_green_space", "is_gated"]):
        df["location_comfort_combo"] = df["location_level"] * (
            df["has_amenities"] + df["has_green_space"] + df["is_gated"]
        )
    else:
        df["location_comfort_combo"] = 0
    return df

@register_feature("location_encoded")
def location_encoded(df):
    if "location" not in df:
        print("⚠️ 缺少 location 字段，跳过 location_encoded")
        return df
    freq_map = df["location"].value_counts().to_dict()
    df["location_encoded"] = df["location"].map(freq_map)
    return df

@register_feature("col_facing_score")
def col_facing_score(df):
    mapping = {
        'East': 4, 'North': 3, 'North - East': 4, 'North - West': 2,
        'South': 0, 'West': 1, 'South - West': 0, 'South - East': 2
    }
    if "facing" not in df:
        print("⚠️ 缺少 facing 字段，跳过 col_facing_score")
        return df
    df["col_facing_score"] = df["facing"].map(mapping).fillna(-1)
    return df

@register_feature("relative_height")
def relative_height(df):
    if "floor_level" not in df or "max_floor" not in df:
        print("⚠️ 缺少 floor_level 或 max_floor，跳过 relative_height")
        return df
    df["relative_height"] = df["floor_level"] / df["max_floor"].replace(0, 1)
    return df


# 48 楼层 + 朝向组合
@register_feature("floor_facing_score")
def floor_facing_score(df):
    if "col_facing_score" not in df or "relative_height" not in df:
        print("⚠️ 缺少 col_facing_score 或 relative_height，跳过 floor_facing_score")
        return df
    df["floor_facing_score"] = df["col_facing_score"] * df["relative_height"]
    return df

# 49
@register_feature("location_ownership_combo")
def location_ownership_combo(df):
    if "location_encoded" in df.columns and "ownership_score" in df.columns:
        df["location_ownership_combo"] = df["location_encoded"] * df["ownership_score"]
    else:
        print("⚠️ 缺少 location_encoded 或 ownership_score，跳过 location_ownership_combo")
    return df

# 50
@register_feature("facing_height_combo")
def facing_height_combo(df):
    if "col_facing_score" in df.columns and "relative_height" in df.columns:
        df["facing_height_combo"] = df["col_facing_score"] * df["relative_height"]
    else:
        print("⚠️ 缺少 col_facing_score 或 relative_height，跳过 facing_height_combo")
    return df

# 51
@register_feature("area_furnishing_combo")
def area_furnishing_combo(df):
    if "Carpet Area" in df.columns and "Furnishing_giving" in df.columns:
        df["area_furnishing_combo"] = df["Carpet Area"] * df["Furnishing_giving"]
    else:
        print("⚠️ 缺少 Carpet Area 或 Furnishing_giving，跳过 area_furnishing_combo")
    return df

#52
@register_feature("quality_score")
def quality_score(df):
    cols = [
        "has_amenities", "has_green_space", "has_proximity",
        "is_affordable", "is_luxury", "is_marketing_strong",
        "is_prime_location", "is_resale", "is_spacious", "is_well_planned"
    ]
    df["quality_score"] = df[cols].sum(axis=1)
    return df


#53
@register_feature("quality_score")
def quality_score(df):
    cols = [
        "has_amenities", "has_green_space", "has_proximity",
        "is_affordable", "is_luxury", "is_marketing_strong",
        "is_prime_location", "is_resale", "is_spacious", "is_well_planned"
    ]
    df["quality_score"] = df[cols].sum(axis=1)
    return df

#54
@register_feature("super_premium_flag")
def super_premium_flag(df):
    cols = [
        "has_amenities", "has_green_space", "has_proximity",
        "is_affordable", "is_luxury", "is_marketing_strong",
        "is_prime_location", "is_resale", "is_spacious", "is_well_planned"
    ]
    df["super_premium_flag"] = df[cols].prod(axis=1)  # 连乘
    return df

#55
@register_feature("log_price")
def log_price(df):
    df_train = df[df["Price (in rupees)"].notna()]
    df_missing = df[df["Price (in rupees)"].isna()]
    # 可选特征（你可以换掉）
    features = [
        "Carpet Area", "floor_level", "max_floor", "location_encoded",
        "is_affordable", "is_luxury", "has_amenities"
    ]
    X = df_train[features]
    y = df_train["Price (in rupees)"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    df.loc[df["Price (in rupees)"].isna(), "Price (in rupees)"] = model.predict(df_missing[features])
    df['log_price'] = np.log1p(df['Price (in rupees)'])
    return df

##其他高级特征工程  补充的话从 39开始补充, 一些数值变量考虑对数化处理之后方便操作
##交叉特征或者其他办法
##df = FEATURE_FUNCTIONS["想继承已经生成的变量"](df)

def add_selected_features(df, features_to_use = None):
    if features_to_use is None:
        features_to_use = list(FEATURE_FUNCTIONS.keys())  # 默认全加，用的时候一定要手动录入！！
    
    for feat in features_to_use:
        if feat in FEATURE_FUNCTIONS:
            df_cleaned_featured = FEATURE_FUNCTIONS[feat](df)
        else:
            print(f"[警告] 未注册的特征函数: {feat}")
    
    return df_cleaned_featured

def list_registered_features():
    return list(FEATURE_FUNCTIONS.keys())
