import pandas as pd
import numpy as np
import re
import random
from sklearn.ensemble import RandomForestClassifier
from utils.config import KEYWORDS

#读取数据
df = pd.read_csv("house_prices.csv")

# main.py 结构类似
from pipline.pipline import pipeline_house_data
from model.train_model import train_model
from model.evaluate import evaluate_model

df = pd.read_csv("data/house_prices.csv")
df_cleaned = pipeline_house_data(df, keywords=KEYWORDS)
print("✅ 数据清洗完成, shape:", df_cleaned.shape)

model, X_test, y_test = train_model(df_cleaned)
evaluate_model(model, X_test, y_test)

