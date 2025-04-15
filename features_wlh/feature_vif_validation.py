import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_multicollinearity(df, features=None, threshold=10, verbose= True):
    """
    检查 DataFrame 中数值型变量的多重共线性（使用 VIF 指标）

    参数:
    df : pd.DataFrame —— 要检查的特征数据
    threshold : float —— 判断共线性是否严重的阈值（通常设为 5 或 10)

    返回:
    vif_df : pd.DataFrame —— 每个变量对应的 VIF 值
    """
    if features:
        df = df[features]  # 只保留指定的列

    # 只保留数值型变量
    numeric_df = df.select_dtypes(include=["number"]).dropna()
    
    # 加常数项用于 statsmodels
    X = sm.add_constant(numeric_df)

    vif_data = {
        "feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }

    vif_df = pd.DataFrame(vif_data)

    print("🧪 方差膨胀因子(VIF)检测结果:")
    print(vif_df[vif_df["VIF"] > threshold])

    #删除特征值逻辑，每轮删除一个最大的VIF
    while True:
        X = sm.add_constant(numeric_df)
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        vif = vif.drop("const")

        max_vif = vif.max()
        if max_vif > threshold:
            drop_feat = vif.idxmax()
            if verbose:
                print(f"⚠️  删除高VIF特征: {drop_feat} (VIF={max_vif:.2f})")
            numeric_df = numeric_df.drop(columns=[drop_feat])
        else:
            break

    VIF_name = vif.index.tolist()
    advanced_features_ultimate = [col for col in VIF_name ]

    return advanced_features_ultimate
