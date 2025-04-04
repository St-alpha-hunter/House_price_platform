import pandas as pd
import numpy as np
import re
import random
from sklearn.ensemble import RandomForestClassifier

class FeatureEngineer:
    def __init__(self, keywords=None, balcony_strategy="leave"):
        self.keywords = keywords
        self.balcony_strategy = balcony_strategy

    def transform(self, df):
        # Step 1: 关键词提取（13个）
        if self.keywords:
            for col, phrases in self.keywords.items():
                pattern = '|'.join(phrases)
                df[col] = df['Description'].str.lower().str.contains(pattern, na=False).astype(int)

        # Step 2: Carpet Area 补全
        df['Carpet Area'] = df['Carpet Area'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        df['Super Area'] = df['Super Area'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
        mask = df['Carpet Area'].isna() & df['Super Area'].notna()
        n = mask.sum()
        random_ratios = np.random.uniform(low=0.75, high=0.88, size=n)
        df.loc[mask, 'Carpet Area'] = df.loc[mask, 'Super Area'] * random_ratios
        df['Carpet Area'] = df['Carpet Area'].fillna(df['Carpet Area'].mode()[0])

        # Step 3: location -> frequency encoding
        location_freq = df['location'].value_counts().to_dict()
        df['location_encoded'] = df['location'].map(location_freq)

        # Step 4: Status
        df['Status'] = df['Status'].fillna('Not allowed to Move')

        # Step 5: Floor 衍生变量
        floor_level_list, max_floor_list = [], []
        is_ground_list, is_basement_list, relative_height_list = [], [], []

        for floor_str in df['Floor']:
            floor_level, max_floor = None, None
            is_basement, is_ground, relative_height = 0, 0, 0

            if isinstance(floor_str, str):
                match = re.match(r'(\w+)\s+out of\s+(\d+)', floor_str.strip())
                if match:
                    raw_level, max_floor = match.groups()
                    max_floor = int(max_floor)
                    if raw_level.isdigit():
                        floor_level = int(raw_level)
                        relative_height = floor_level / max_floor
                    elif 'ground' in raw_level.lower():
                        floor_level = 0
                        is_ground = 1
                    elif 'basement' in raw_level.lower():
                        floor_level = -1
                        is_basement = 1
                elif 'ground' in floor_str.lower():
                    floor_level, is_ground = 0, 1
                    max_floor = 0
                elif 'basement' in floor_str.lower():
                    floor_level, is_basement = -1, 1
                    max_floor = 0
            else:
                floor_level = random.randint(5, 20)
                max_floor = random.randint(5, 30)
                is_ground = random.randint(0, 1)
                is_basement = 0 if is_ground else random.randint(0, 1)
                relative_height = random.random()

            floor_level_list.append(floor_level)
            max_floor_list.append(max_floor)
            is_ground_list.append(is_ground)
            is_basement_list.append(is_basement)
            relative_height_list.append(relative_height)

        df['floor_level'] = floor_level_list
        df['max_floor'] = max_floor_list
        df['is_ground'] = is_ground_list
        df['is_basement'] = is_basement_list
        df['relative_height'] = relative_height_list

        # Step 6~7 Transaction & Furnishing
        df['Transaction'] = df['Transaction'].fillna(df['Transaction'].mode()[0])
        df['Furnishing'] = df['Furnishing'].fillna(df['Furnishing'].mode()[0])

        # Step 8 Facing
        facing_score_map = {'East': 4, 'North': 3, 'North - East': 4, 'North - West': 2,
                            'South': 0, 'West': 1, 'South - West': 0, 'South - East': 2}
        df['col_facing_score'] = df['facing'].apply(lambda x: facing_score_map.get(x, -1))
        df.drop(columns=['facing'], inplace=True)

        # Step 9 Overlooking
        def score_overlook(x):
            score = 0
            if isinstance(x, str):
                if 'Garden/Park' in x: score += 1
                if 'Pool' in x: score += 1
                if 'Main Road' in x or 'Not Available' in x: score -= 1
                if 'Garden/Park' in x and 'Pool' in x: score += 1
            else:
                score = -1
            return score
        df['col_overlooking_score'] = df['overlooking'].apply(score_overlook)
        df.drop(columns=['overlooking'], inplace=True)

        # Step 10 Society from Title
        def extract_society(title):
            if pd.isna(title): return None
            match = re.search(r'sale in (.+)', title.lower()) or re.search(r'sale (.+)', title.lower())
            return match.group(1).strip().title() if match else None

        def normalize_society_name(name):
            if pd.isna(name): return np.nan
            name = re.sub(r'[^a-z0-9 ]', '', str(name).lower().strip())
            return ' '.join(sorted(name.split()))

        df['Society'] = df.apply(
            lambda row: extract_society(row['Title']) if pd.isna(row['Society']) else row['Society'], axis=1)
        df['society_clean'] = df['Society'].apply(normalize_society_name)
        df['society_freq'] = df['society_clean'].map(df['society_clean'].value_counts().to_dict())
        df['society_level'] = pd.qcut(df['society_freq'], q=5, labels=False, duplicates='drop')
        df.drop(columns=['society_freq', 'society_clean', 'Society'], inplace=True)

        # Step 11 Bathroom
        def map_bathroom(x):
            try:
                x = float(x)
                return 1 if x <= 1 else (2 if x == 2 else 3 if x == 3 else 4 if x in [4, 5] else 5)
            except:
                return 5
        df['col_Bathroom_score'] = df['Bathroom'].apply(map_bathroom)
        df.drop(columns=['Bathroom'], inplace=True)

        # Step 12 Balcony - 留给策略控制
        if self.balcony_strategy == "zero":
            df['Balcony'] = df['Balcony'].fillna(0)
        elif self.balcony_strategy == "median":
            df['Balcony'] = df['Balcony'].fillna(df['Balcony'].median())
        elif self.balcony_strategy == "drop":
            df = df[df['Balcony'].notna()]
        # 否则 leave 保持缺失

        # Step 13 Car Parking
        def car_score(x):
            if pd.isna(x): return 0
            elif 'Covered' in x: return 2
            elif 'Open' in x: return 1
            else: return 0
        df['col_car_parking_score'] = df['Car Parking'].apply(car_score)
        df.drop(columns=['Car Parking'], inplace=True)

        # Step 14 Ownership 补全 + 随机森林预测
        def extract_ownership(desc):
            if pd.isna(desc): return None
            desc = desc.lower()
            if 'freehold' in desc: return 'Freehold'
            elif 'leasehold' in desc: return 'Leasehold'
            elif 'poa' in desc or 'power of attorney' in desc: return 'Power Of Attorney'
            elif 'co-operative' in desc: return 'Co-operative Society'
            return None

        df['Ownership'] = df.apply(
            lambda row: extract_ownership(row['Description']) if pd.isna(row['Ownership']) else row['Ownership'], axis=1)

        ownership_map = {'Freehold': 3, 'Leasehold': 2, 'Co-operative Society': 1, 'Power Of Attorney': 0}
        df['ownership_score'] = df['Ownership'].map(ownership_map)

        features = ['Carpet Area', 'col_facing_score', 'location_encoded', 'floor_level', 'max_floor',
                    'is_ground', 'is_resale', 'is_gated', 'has_amenities', 'has_green_space']

        train_data = df[df['ownership_score'].notna()]
        test_data = df[df['ownership_score'].isna()]
        if not train_data.empty and not test_data.empty:
            model = RandomForestClassifier()
            model.fit(train_data[features], train_data['Ownership'])
            df.loc[df['ownership_score'].isna(), 'Ownership'] = model.predict(test_data[features])

        df['Ownership'] = df['Ownership'].fillna('Unknown')
        ownership_map['Unknown'] = -1
        df['ownership_score'] = df['Ownership'].map(ownership_map)

        # Step 15 Price字段清洗
        def convert_amount_to_rupiah(text):
            try:
                text = str(text).lower().replace(',', '').strip()
                match = re.search(r'([\d\.]+)\s*([a-zA-Z]+)?', text)
                if match:
                    num, unit = match.groups()
                    num = float(num)
                    if unit in ['lac', 'lacs']: return int(num * 1e5)
                    elif unit in ['cr', 'crore']: return int(num * 1e7)
                    elif unit in ['thousand', 'k']: return int(num * 1e3)
                    else: return int(num)
                return np.nan
            except:
                return np.nan

        df['Amount_clean'] = df['Amount(in rupees)'].apply(convert_amount_to_rupiah)
        df = df[df['Amount(in rupees)'] != 'Call for Price'].copy()

        # Step 16 删除无用列
        df = df.drop(columns=["Amount(in rupees)", "Dimensions", "Plot Area", "Title", "Description",
                              "Floor", "Super Area", "Ownership", "location"], errors='ignore')

        # Step 17 编码分类特征
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes

        # Step 18 填补所有NaN（中位数）
        df = df.fillna(df.median(numeric_only=True))

        return df
