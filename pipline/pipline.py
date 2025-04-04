import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
import random

def pipeline_house_data(df,keywords,col_Title = "Title",
                           col_Description = "Description",
                           col_Amount = "Amount(in rupees)",
                           col_Price = "Price (in rupees)",
                           col_location = "location",
                           col_Carpet_Area = "Carpet Area",
                           col_Status = "Status",
                           col_Floor = "Floor",
                           col_Transaction = "Transaction",
                           col_Furnishing = "Furnishing",
                           col_facing = "facing",
                           col_overlooking = "overlooking",
                           col_Society = "Society",
                           col_Bathroom = "Bathroom",
                           col_Balcony = "Balcony",
                           col_Car_Parking = "Car Parking",
                           col_Ownership = "Ownership",
                           col_Super_Area = "Super Area",
                           col_Dimensions = "Dimensions",
                           col_Plot_Area = "Plot Area",
                           seed=42, ratio_range=(0.75, 0.88)
                          ):
    
#step1---从Description中提取  
   #'is_prime_location', 'has_proximity', 'is_well_planned','is_new',
   #'is_resale','is_affordable','is_deal', 'is_spacious','is_luxury',
   #'is_gated','has_amenities','has_green_space','is_marketing_strong' 这13个特征
   # 批量提取为二值字段
    for col, phrases in keywords.items():
       pattern = '|'.join(phrases)
       df[col] = df['Description'].str.lower().str.contains(pattern, na=False).astype(int)

#step2---处理Carpet_Area, Super_Area, 用Super_Area来填充Carpet_Area, 其余0.5%用众数填充,再舍去Super_Area
    df[col_Carpet_Area] = df[col_Carpet_Area].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    df[col_Super_Area] = df[col_Super_Area].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
   # 创建填补掩码：Carpet 缺失但 Super 有值
    mask = df[col_Carpet_Area].isna() & df[col_Super_Area].notna()
    # 随机生成比例
    n = mask.sum()
    random_ratios = np.random.uniform(low=ratio_range[0], high=ratio_range[1], size=n)
    # 计算填补值
    df.loc[mask, col_Carpet_Area] = df.loc[mask, col_Super_Area] * random_ratios
    df[col_Carpet_Area] = df[col_Carpet_Area].fillna(df[col_Carpet_Area].mode()[0])

#step3---处理location,统计出各个location的频度
    location_freq = df[col_location].value_counts().to_dict()
    df['location_encoded'] = df[col_location].map(location_freq)

#step4---处理Status, 原来数据只有ready to move，和缺失值，所以把nan 赋值成 not allowed to move
    df[col_Status] = df[col_Status].fillna('Not allowed to Move')

#step5--处理floor, 由floor 衍生出4个布尔变量，1个相对高度变量
    floor_level_list = []
    max_floor_list = [] 
    is_ground_list = []
    is_basement_list = []
    relative_height_list = []

    is_basement = 0   # ✅ 提前初始化
    is_ground = 0
    max_floor = None
    floor_level = None
    relative_height = None
    for floor_str in df[col_Floor]:
        if isinstance(floor_str, str):
            floor_str = floor_str.strip()
            # 匹配 '3 out of 10' 类似格式
            match = re.match(r'(\w+)\s+out of\s+(\d+)', floor_str)
            if match:
                raw_level, max_floor = match.groups()
                max_floor = int(max_floor)  #衍生1
                
                if raw_level.isdigit():
                    floor_level = int(raw_level)
                    relative_height = floor_level/max_floor #衍生2
                    
                elif 'ground' in raw_level.lower():
                   floor_level = 0 
                   is_ground = 1              #衍生3
                   is_basement = 0
                   relative_height = 0

                elif 'basement' in raw_level.lower():
                   floor_level = -1
                   is_basement = 1            #衍生4
                   is_ground = 0
                   relative_height = 0
 
                           
            else:
                # 没匹配上，比如只有 'Ground' 之类的
                if 'ground' in floor_str.lower():
                   floor_level = 0
                   is_ground = 1
                   is_basement = 0
                   max_floor = 0
                   relative_height = 0

                    
                elif 'basement' in floor_str.lower():
                   floor_level = -1
                   is_basement = 1
                   is_ground = 0
                   max_floor = 0
                   relative_height = 0

        else:
                # 没匹配任何格式，给出合理随机值
            floor_level = random.randint(5, 20)
            max_floor = random.randint(5, 30)
                
                # 随机地给 is_ground 或 is_basement = 1，但不能同时为 1
            is_ground = random.randint(0, 1)
            is_basement = 0 if is_ground == 1 else random.randint(0, 1)
            relative_height = random.random()

                    

        floor_level_list.append(floor_level)
        is_basement_list.append(is_basement)
        is_ground_list.append(is_ground)
        max_floor_list.append(max_floor)
        relative_height_list.append(relative_height)

    # 加回 DataFrame
    df['floor_level'] = floor_level_list
    df['max_floor'] = max_floor_list
    df['is_ground'] = is_ground_list
    df['is_basement'] = is_basement_list
    df['relative_height'] = relative_height_list

#step6--处理Transaction
    most_common_1 = df[col_Transaction].mode()[0]
    df[col_Transaction] = df[col_Transaction].fillna(most_common_1)

#step7--处理furnishing
    most_common_2 = df[col_Furnishing].mode()[0]
    df[col_Furnishing] = df[col_Furnishing].fillna(most_common_2)

#step8--处理facing
    facing_score_map = {
         'East': 4,
         'North': 3,
         'North - East': 4,
         'North - West': 2,
         'South': 0,
          'West': 1,
          'South - West': 0,
          'South - East': 2
    }
    col_facing_score = []
    col_facing_score = df['facing'].map(facing_score_map)
    col_facing_score = df[col_facing].apply(lambda x: facing_score_map.get(x, -1))
    df["col_facing_score"] = col_facing_score
    df.drop(columns=[col_facing], inplace=True)  # ✅ 正确


#step9--处理overlooking,引入评分机制来处理
    col_overlooking_score = []
    for i in df[col_overlooking]:
        score = 0
        if isinstance(i, str):
            if 'Garden/Park' in i:
                score += 1
            if 'Pool' in i:
                score += 1
            if 'Main Road' in i:
                score -= 1
            if 'Not Available' in i:
                score -= 1
            if 'Garden/Park' in i and 'Pool' in i:
                score += 1  # 额外加分
        else:
            score = -1
        col_overlooking_score.append(score)
    df["col_overlooking_score"] = col_overlooking_score
    df.drop(columns = [col_overlooking], inplace = True)


#step 10
    def extract_society_from_title(title):
        if pd.isna(title):
            return None
        
        title = title.lower()
        
        # 优先匹配 'sale in'
        match_in = re.search(r'sale in (.+)', title)
        if match_in:
            return match_in.group(1).strip().title()  # 保留大写格式
        
        # 次级匹配 'sale'
        match = re.search(r'sale (.+)', title)
        if match:
            return match.group(1).strip().title()
        
        return None
    
    # 对缺失的 society 进行填补
    df['Society'] = df.apply(
    lambda row: extract_society_from_title(row['Title']) if pd.isna(row['Society']) else row['Society'],
    axis=1
    )

    def normalize_society_name(name):
        if pd.isna(name):
            return np.nan
        name = str(name).lower().strip()
        name = re.sub(r'[^a-z0-9 ]', '', name)  # 去掉标点
        words = name.split()
        words.sort()  # 排序词汇（更激进）
        return ' '.join(words)
    
    df['society_clean'] = df['Society'].apply(normalize_society_name)
    society_freq = df['society_clean'].value_counts().to_dict()
    
    df['society_freq'] = df['society_clean'].map(society_freq)
    
    
    df['society_level'] = pd.qcut(df['society_freq'], q=5, labels=False, duplicates='drop')
    df = df.drop(columns=['society_freq','society_clean',"Society"])
    

#step11--处理Bathroom
    col_Bathroom_score = []
    for i in df[col_Bathroom]:
        try:
            x = float(i)
            if x <= 1: 
                x = 1
            elif x == 2: x = 2
            elif x == 3: x = 3
            elif x == 4 or x == 5: x = 4
            else: x == 5
        except:
            x = 5
        col_Bathroom_score.append(x)
    # 确保长度一致
    assert len(col_Bathroom_score) == len(df)
    df["col_Bathroom_score"] = col_Bathroom_score
    df.drop(columns=col_Bathroom)
#step12--处理Balcony,25%缺失，后续根据训练思路来处理          
#step13--处理car-parking
    col_car_parking_score= []
    for j in df[col_Car_Parking]:
        if pd.isna(j):
              j = 0
        elif 'Covered' in j:
              j = 2
        elif 'Open' in j:
              j = 1
        else:
             j = 0
        col_car_parking_score.append(j)
    df["col_car_parking_score"] = col_car_parking_score
    df.drop(columns = col_Car_Parking)


#step14--处理Ownership
    features = [
        'Carpet Area', 'col_facing_score', 'location_encoded',
        'floor_level', 'max_floor', 'is_ground', 'is_resale',
        'is_gated', 'has_amenities', 'has_green_space'
    ]
    # StepX — 清洗并补全 Ownership 字段
    def extract_ownership(desc):
        if pd.isna(desc): return None
        desc = desc.lower()
        if 'freehold' in desc:
            return 'Freehold'
        elif 'leasehold' in desc:
            return 'Leasehold'
        elif 'power of attorney' in desc or 'poa' in desc:
            return 'Power Of Attorney'
        elif 'co-operative society' in desc or 'cs' in desc:
            return 'Co-operative Society'
        else:
            return None
    
    # StepX.1 — 优先从 Description 中提取缺失值
    df[col_Ownership] = df.apply(
        lambda row: extract_ownership(row[col_Description]) if pd.isna(row[col_Ownership]) else row[col_Ownership],
        axis=1
    )
    # StepX.2 — 映射为得分字段（便于建模）
    ownership_map = {
        'Freehold': 3,
        'Leasehold': 2,
        'Co-operative Society': 1,
        'Power Of Attorney': 0
    }
    df['ownership_score'] = df[col_Ownership].map(ownership_map)
    
    # StepX.3 — 用随机森林预测剩余缺失的 Ownership
    from sklearn.ensemble import RandomForestClassifier
    # 构造训练和测试集（仅对 ownership_score 是缺失的行进行预测）
    train_data = df[df['ownership_score'].notna()]
    test_data = df[df['ownership_score'].isna()]
    # 注意：feature 应该在 pipeline 外部传入或你提前在函数中定义好
    X_train = train_data[features]
    y_train = train_data[col_Ownership]
    X_test = test_data[features]
    # 训练并预测
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    df.loc[df['ownership_score'].isna(), col_Ownership] = model.predict(X_test)
    # StepX.4 — 再次映射成得分（把新预测的也加进来）
    df[col_Ownership] = df[col_Ownership].fillna('Unknown')  # 万一有漏
    ownership_map_with_unknown = {**ownership_map, 'Unknown': -1}
    df['ownership_score'] = df[col_Ownership].map(ownership_map_with_unknown)
    
#step15-处理好Amount
    # 解释： \d[\d.,]* 匹配数字部分，([a-zA-Z]+) 捕获后缀
    #suffixes = amount_series.str.extract(r'[\d,\.]+\s*([a-zA-Z]+)?')[0].fillna('No Unit')
    # 9684个是竞价交易，所以剔除这些数据
    def convert_amount_to_rupiah(text):
        try:
            text = str(text).lower().replace(',', '').strip()
    
            # 匹配金额和单位，例如 "1.5 cr", "25 lac"
            match = re.search(r'([\d\.]+)\s*([a-zA-Z]+)?', text)
            if match:
                num, unit = match.groups()
                num = float(num)
    
                if unit in ['lac', 'lacs']:
                    return int(num * 1e5)
                elif unit in ['cr', 'crore']:
                    return int(num * 1e7)
                elif unit in ['thousand', 'k']:
                    return int(num * 1e3)
                else:
                    return int(num)  # 无单位，直接用
    
            return np.nan  # 没匹配成功
        except:
            return np.nan
    df['Amount_clean'] = df['Amount(in rupees)'].apply(convert_amount_to_rupiah)
    # 9684个是竞价交易，所以剔除这些数
    df = df[df['Amount(in rupees)'] != 'Call for Price'].copy()

    

#step-16删除完全空缺值, 辅助列
    df = df.drop(columns = ["Amount(in rupees)","Dimensions","Plot Area","Title","Description","Floor","Super Area", "Car Parking", "Ownership", "Bathroom", "location"])

# 统一对所有分类列进行编码处理
    categorical_cols = ['Status', 'Transaction', 'Furnishing']
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

# ⭐ 新增：统一处理其他可能的字符串列
    remaining_str_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in remaining_str_cols:
        df[col] = df[col].astype('category').cat.codes

# ✅ 统一填充所有NaN，推荐使用中位数填充
    df = df.fillna(df.median(numeric_only=True))

    return df