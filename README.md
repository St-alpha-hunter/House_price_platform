#### PROJECT NAME ####
-----Machine Learning Regression Pipeline

#### PROJECT DESCRIPTION ####
This project makes a machine-learning platform which is used to predict the price of house. Supporting Visualization Analysis,washing data by pipeline,eastbalishing the features you like, selecting useful features,searching best params by grid, cross_validition. 

the models users can use on the platform by now:
- RandomForestRegressor
- XGBRegressor
- LGBMRegressor
- CatBoostRegressor



## PROJECT STRUCTURE ##

```
HOUSE_PRICE_PLATFORM/
├── advanced_modify/           # 高级功能模块 / advanced features
│   ├── cross_validation.py    # 多指标交叉验证工具 / cross-validation tools
│   └── grid_search_tool.py    # 网格搜索模块 / grid search module (pending)
├── dashboard/
│   └── powerbi_dashboard.pbix # PowerBI 可视化文件 / PowerBI dashboard (pending)
├── data/                      # 原始 & 处理后数据 / raw & processed data
│   └── house_prices_cleaned.csv
├── features_wlh/              # 特征工程模块 / feature engineering
│   ├── feature_analysis.py
│   ├── feature_selector.py
│   └── FeatureDeepAnalysis.py
├── model/                     # 模型相关逻辑 / model logic
│   ├── evaluate.py            # 模型评估方法 / model evaluation
│   └── train_model.py         # 模型训练主函数 / training function
├── models_saved/              # 训练好的模型保存目录 / saved models
├── notebooks/
│   └── practice.ipynb         # 交互式实验笔记本 / experiment notebook
├── pipeline/                  # 项目主流程模块 / main pipeline module
├── report/                    # 报告生成模块 / report generator (extendable)
├── utils/                     # 工具包 / utility functions
│   ├── config.py              # 模型参数配置中心 / model config center
│   ├── features_name.text     # 特征名称列表 / feature name list
│   └── path_helper.py         # 路径工具 / path helper
├── main.py                    # 主程序入口 / project entry point
├── README.md                  # 项目说明文档 / project instructions
├── requirement.txt            # 环境依赖列表 / environment requirements
└── .gitignore                 # Git 忽略规则 / git ignore rules
```




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

###### Instruction Manual ######
1. The platform have two parts of training.  The experiment procedure in notebook called platform E, the informal procedure in main.py called Platform A
2. platform E is more convenient and flexible. Although we set some solid parts in notebook to assist you, you can try Exploratory Data Analysis.
3. After you fininsh your training in platform E and get the best params, you should record the best params in config.py
4. After that you can run platform A (Hint: the features and model, you choose in E and A must be the same),the report and model can be recorded automatically.

###### To Do ######
1. do some front-end development (using steamlit)
2. publish this project on some Cloud-platform
3. develop logger module to record the experiments 
4. others



###### Author Information ######
Author: leader:Lihao Wang 
        member:Xinping Luo /Shuqi Deng/ Ziqi Zhong
School/Team: Nottingham 
