#### PROJECT NAME ####
-----Machine Learning Regression Pipeline

#### PROJECT DESCRIPTION ####
This project makes a machine-learning platform which is used to predict the price of house. Supporting Visualization Analysis,washing data by pipeline,eastbalishing the features you like, selecting useful features,searching best params by grid, cross_validition. 

the models users can use on the platform by now:
- RandomForestRegressor
- XGBRegressor
- LGBMRegressor
- CatBoostRegressor


###### PROJECT STRUCTURE ######
HOUSE_PRICE_PLATFORM/
│
├── advanced_modify/              # 高级功能模块
│   ├── cross_validation.py       # 多指标交叉验证工具
│   ├── grid_search_tool.py       # 网格搜索模块   please waiting
│   ├── pca_transformer.py        # PCA 降维模块   please waiting
│
├── dashboard/
│   └── powerbi_dashboard.pbix    # PowerBI 可视化文件  please waiting
│
├── data/                         # 原始 & 处理后数据
│   ├── house_prices.csv
│   └── processed_data/
│       └── house_prices_cleaned.csv
│
├── features_wlh/                 # 特征工程模块 
│   ├── feature_analysis.py
│   ├── feature_selector.py
│   ├── feature_vif_validation.py
│   ├── FeatureDeepAnalysis.py
│   └── features_wlh.py
│
├── model/                        # 模型相关逻辑
│   ├── evaluate.py               # 模型评估方法
│   └── train_model.py            # 模型训练主函数
│
├── models_saved/                 # 训练好的模型保存目录
│
├── notebooks/
│   └── practice.ipynb            # 交互式实验笔记本
│
├── pipline/                      # 项目主流程模块
│   └── pipline.py
│
├── report/                       # 报告生成模块（可扩展）
│
├── utils/                        # 工具包
│   ├── config.py                 # 模型参数配置中心
│   ├── features_name.text        # 特征名称列表
│   └── path_helper.py            # 路径工具
│
├── main.py                       # 主程序入口
├── README.md                     # 项目说明文档
├── requirement.txt               # 环境依赖列表
└── .gitignore                    # Git 忽略规则




###### Installment Environment ####
The project was developed and tested in the following environment:

- Python 3.10+
- Jupyter Notebook / Python script
- OS: Windows 10 / macOS / Linux

1.Clone the Repository
  git clone https://github.com/St-alpha-hunter/House_price_platform
  cd House_price_platform

2.Create Virtual Environment (Recommended)
  #Using venv
  python -m venv venv
  source venv/bin/activate   # On Windows use: venv\Scripts\activate

3.pip install -r requirements.txt

4.python main.py


###### To Do ######
1. do some front-end development (using steamlit)
2. publish this project on some Cloud-platform
3. develop logger module to record the experiments 
4. others



###### Author Information ######
Author: leader:Lihao Wang 
        member:Xinping Luo /Shuqi Deng/ Ziqi Zhong
School/Team: Nottingham 
