import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_correlation(df, features=None, threshold=0.85):
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
