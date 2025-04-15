from sklearn.model_selection import cross_validate, KFold
import pandas as pd
import numpy as np

def enhanced_cross_validate(model, features=None, target_col = None, cv=5, verbose=True, return_df=False):
    """
    支持 MAE、MSE、R² 的交叉验证函数
    """

    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error',
        'R2': 'r2'
    }

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_validate(model, features, target_col, cv=kfold, scoring=scoring, return_train_score=False)

    # 转为 DataFrame 方便后处理
    df_scores = pd.DataFrame(scores)
    
    if verbose:
        print(f"📊 {cv}-折交叉验证结果:")
        for metric in ['MAE', 'MSE', 'R2']:
            test_col = f'test_{metric}'
            mean = df_scores[test_col].mean()
            std = df_scores[test_col].std()
            if metric != 'R2':  # MAE, MSE 是负的
                mean, std = -mean, std
            print(f"🔹 {metric:<4}: {mean:.4f} ± {std:.4f}")

    if return_df:
        return df_scores
    else:
        return {metric: df_scores[f'test_{metric}'] for metric in scoring}

##指定评估标准（回归模型最常见如 MAE、RMSE、R²）
##verbose, 如果为 True，就会打印每一折的得分和均值±标准差