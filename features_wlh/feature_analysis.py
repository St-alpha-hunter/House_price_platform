import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_correlation(df, features=None, threshold=0.85,verbose=True):
    """
    显示特征间的相关性热力图，并标记高相关对。
    """
    if features is None:
        features = df.select_dtypes(include='number').columns

    corr_matrix = df[features].corr()

    # 热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("📊 Feature Correlation Heatmap")
    plt.show()

    # 输出高度相关的特征对
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                score = corr_matrix.iloc[i, j]
                high_corr.append((col1, col2, round(score, 3)))

    if high_corr:
        print("⚠️ 高度相关的特征对（|相关性| > {})：".format(threshold))
        for col1, col2, score in high_corr:
            print(f"{col1} & {col2} → 相关系数: {score}")
    else:
        print("✅ 没有检测到高度相关的特征对。")


    #自动处理高相关特征值对
    #取的是 相关矩阵的上三角（即只看每对组合一次，跳过对称部分）,靠右的列优先被删，靠左的列优先保留
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if verbose:
        print(f"🔍 检测到 {len(to_drop)} 个高相关特征将被删除（阈值：{threshold}）：")
        print(to_drop)
 
    advanced_feature = [col for col in features if col not in to_drop]

    return advanced_feature


