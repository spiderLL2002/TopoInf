import os
import pandas as pd

# 确保脚本运行的当前工作目录正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 文件路径
base_dir = './output1/'
datasets = ["cora" , "citeseer" , "pubmed"]
files = ['GCN.csv', 'SGC.csv', 'APPNP.csv']

# 对 CSV 文件排序并覆盖保存
def sort_and_save(file_path):
    if os.path.exists(file_path):
        # 读取数据
        data = pd.read_csv(file_path)
        
        # 对相同 ratio 的 test_acc_mean 取平均值，并排序
        data = data.groupby('ratio', as_index=False)['test_acc_mean'].mean()
        data = data.sort_values(by='ratio')
        
        # 覆盖保存文件
        data.to_csv(file_path, index=False)
        print(f"Sorted and saved: {file_path}")
    else:
        print(f"File not found: {file_path}")

# 对每个文件进行处理
for dataset in datasets  : 
    for file_name in files:
        sort_and_save(os.path.join(base_dir,dataset , file_name))
