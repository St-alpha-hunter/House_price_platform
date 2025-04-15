import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def feature_selection_by_k(features, target_col, max_k=20, rank_features=10, model_cls=RandomForestRegressor):
 
    results = []

    # Step 1: 用过滤法选 top max_k 个初始特征
    filter_selector = SelectKBest(score_func=f_regression, k=max_k)
    X_filtered = filter_selector.fit_transform(features, target_col)
    filtered_cols = features.columns[filter_selector.get_support()]
    X_filtered_df = features[filtered_cols]

    # Step 2: 遍历 k，嵌入法评估每组前 k 特征的贡献
    for k in range(10, max_k + 1):
        X_k = X_filtered_df.iloc[:, :k]
        model = model_cls(n_estimators=50, random_state=42)
        model.fit(X_k, target_col)

        importances = model.feature_importances_
        num_features_to_rank = min(rank_features, k)
        top_k_idx = np.argsort(importances)[::-1][:num_features_to_rank]
        top_k_features = X_k.columns[top_k_idx]
        X_top_k = X_k[top_k_features]  # optional

        results.append((k, top_k_features.tolist()))

    results_df = pd.DataFrame(results, columns=["num_features", "top_features"])
    return results_df


def select_final_top_features(features, target_col, max_k=20, top_k=15, model_cls=RandomForestRegressor):
    from sklearn.feature_selection import SelectKBest, f_regression
    import numpy as np

    # Step 1: 过滤法初筛
    filter_selector = SelectKBest(score_func=f_regression, k=max_k)
    X_filtered = filter_selector.fit_transform(features, target_col)
    filtered_cols = features.columns[filter_selector.get_support()]
    X_filtered_df = features[filtered_cols]

    # Step 2: 嵌入法评估特征重要性
    model = model_cls(n_estimators=100, random_state=42)
    model.fit(X_filtered_df, target_col)

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:top_k]
    top_features = X_filtered_df.columns[top_idx]
    top_features = top_features.tolist()

    return top_features
