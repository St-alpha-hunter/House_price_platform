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

    df["normal_price"] = (df["Price (in rupees)"] - df["Price (in rupees)"].mean())/(df["Price (in rupees)"].max() - df["Price (in rupees)"].min())
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
    map_transaction = {'New Property':2, 'Rent/Lease':1, "other":0, "resale":-1}
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
@register_feature("facing_giviing")
def facing_giving(df):
    df["facing_giving"] = df["col_facing_score"]
    return df

#12
@register_feature("facing_hot")
def facing_hot(df):
    df["facing_hot"] = pd.get_dummies(df["col_facing_score"], prefix="col_facing_score")
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
    return df

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
def Car_parking(df):
    df["Car_parking"] = df ["col_car_parking_score"]
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
    map_hot = {'0':4, '1':3, '2':2, '3':1, '4':0}
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

##其他高级特征工程  补充的话从 39开始补充, 一些数值变量考虑对数化处理之后方便操作
##交叉特征或者其他办法


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
