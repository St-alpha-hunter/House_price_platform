from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from xgboost import XGBRegressor, XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class FeatureDeepAnalysis:
    def __init__(self,df,features=None,model_cls=None,target_col="Amount_clean"):
        self.df = df
        self.features = features
        self.model_cls = model_cls
        self.target_col = target_col

    ##分析特征值重要性
    def plot_feature_importance(self):
        if self.model_cls == XGBRegressor:
           model = XGBRegressor(n_estimators=100, random_state=42)
           model.fit(self.df[self.features], self.df[self.target_col])
           plot_importance(model, max_num_features=15)
           plt.title("Top 15 Feature Importance (XGB)")
           plt.show()

        elif self.model_cls == LGBMRegressor:
            model = LGBMRegressor(n_estimators=100, random_state=42)
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.feature_importances_
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.nlargest(15).plot(kind='barh')
            plt.title("Top 15 Feature Importance (LGBM)")
            plt.show()

        elif self.model_cls == CatBoostRegressor:
            model = CatBoostRegressor(verbose=0, random_state=42)
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.get_feature_importance()
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.nlargest(15).plot(kind='barh')
            plt.title("Top 15 Feature Importance (CatBoost)")
            plt.show()

        elif self.model_cls == RandomForestRegressor:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(self.df[self.features], self.df[self.target_col])
            importance = model.feature_importances_
            feat_imp = pd.Series(importance, index=self.features)
            feat_imp.nlargest(15).plot(kind='barh')
            plt.title("Top 15 Feature Importance (Random Forest)")
            plt.show()

    #一键画图
    def plot_feature_distribution(self):
        for col in self.features:
            plt.figure(figsize=(6, 3))
            sns.set_style("whitegrid")
            sns.histplot(self.df[col], kde=True)
            plt.title(f"{col} (skew = {self.df[col].skew():.2f})")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.show()
            



    #和目标的相关性
    def plot_feature_vs_target(self):
        for col in self.features:
            plt.figure(figsize=(6, 3))
            sns.set_style("whitegrid")
            sns.scatterplot(data=self.df, x=col, y=self.target_col, alpha=0.3)
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.title(f"{col} vs {self.target_col}")
            plt.tight_layout()
            plt.show()
