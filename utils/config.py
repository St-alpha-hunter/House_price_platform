# ========================
# 原始列名配置（建议统一管理）
# ========================

COL_TITLE = "Title"
COL_DESCRIPTION = "Description"

COL_AMOUNT = "Amount(in rupees)"
COL_PRICE = "Price (in rupees)"
COL_LOCATION = "location"

COL_CARPET_AREA = "Carpet Area"
COL_SUPER_AREA = "Super Area"
COL_PLOT_AREA = "Plot Area"
COL_DIMENSIONS = "Dimensions"

COL_STATUS = "Status"
COL_TRANSACTION = "Transaction"
COL_FURNISHING = "Furnishing"
COL_FLOOR = "Floor"
COL_BATHROOM = "Bathroom"
COL_BALCONY = "Balcony"
COL_CAR_PARKING = "Car Parking"
COL_OWNERSHIP = "Ownership"
COL_SOCIETY = "Society"

COL_FACING = "facing"
COL_OVERLOOKING = "overlooking"


KEYWORDS = {
'is_prime_location': ['prime location', 'central', 'in the heart of'],
    'has_proximity': ['close to', 'near', 'adjacent to'],
    'is_well_planned': ['well planned', 'meticulously planned', 'creatively constructed'],
    'is_new': ['brand new'],
    'is_resale': ['resale property'],
    'is_affordable': ['affordable', 'reasonable price', 'economic pricing'],
    'is_deal': ['great deal', 'unbelievable price', 'bargain'],
    'is_spacious': ['spacious'],
    'is_luxury': ['luxurious', 'premium', 'desirable'],
    'is_gated': ['gated community', 'society', 'township', 'complex'],
    'has_amenities': ['clubhouse', 'gym', 'swimming pool'],
    'has_green_space': ['green zone', 'open space'],
    'is_marketing_strong': ['perfect property', 'don’t miss', 'must see']
}

# ========================
# 随机种子与比例范围
# ========================

RANDOM_SEED = 42
RATIO_RANGE = (0.75, 0.88)

# ========================
# 模型参数
# ========================
# config.py
# config.py
model_configs = {
    "RandomForestRegressor": {
        "max_depth": None,
        "min_samples_split": 2,
        "n_estimators": 200
    },
    
    "XGBRegressor": {
        # 可添加参数，比如：
        # "n_estimators": 100,
        # "learning_rate": 0.1,
    },

    "LGBMRegressor": {
        # "num_leaves": 31,
        # "n_estimators": 100,
    },

    "CatBoostRegressor": {
        # "depth": 6,
        # "iterations": 200,
    }
}
