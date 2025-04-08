import os

def get_project_root():
    try:
        # 脚本运行
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        # Notebook 环境
        return os.path.abspath(os.path.join(os.getcwd(), ".."))

def get_data_path(filename):
    return os.path.join(get_project_root(), "data", filename)
