import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 确保脚本运行的当前工作目录正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 文件路径
base_dir = './output1/'
datasets = ["cora", "citeseer", "pubmed"]
files = ['GCN.csv', 'SGC.csv', 'APPNP.csv']

# 数据预处理函数 
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # 对 ratio 列进行排序，并对重复的 ratio 取平均值
    data = data.groupby('ratio', as_index=False)['test_acc_mean'].mean()
    data = data.sort_values(by='ratio')
    return data

# 绘制函数
def plot_dataset(dataset):
    plt.figure(figsize=(12, 8))

    colors = sns.color_palette("Set2", n_colors=len(files))
    markers = ['o', 's', '^']
    labels = ['GCN', 'SGC', 'APPNP']

    for file, color, label, marker in zip(files, colors, labels, markers):
        file_path = os.path.join(base_dir, dataset, file)
        data = preprocess_data(file_path)

        # 绘制原始数据点
        plt.scatter(data['ratio'], data['test_acc_mean'], label=label, color=color, s=20, marker=marker, alpha=0.5)
        
        # 绘制连线 - 增加线条粗细
        plt.plot(data['ratio'], data['test_acc_mean'], color=color, linestyle='--', linewidth=6, alpha=0.8)

    plt.title(f'{dataset.capitalize()}: Test Accuracy vs Ratio', fontsize=20)

    plt.grid(alpha=0.3)

    # 添加图例并增加字体大小
    plt.legend(fontsize=24)

    # 设置刻度标签字体大小
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # 确保输出目录存在
    output_dir = './vis1/'
    os.makedirs(output_dir, exist_ok=True)  # 自动创建目录（包括父目录）

    # 保存图片
    output_file = os.path.join(output_dir, f'{dataset}.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Visualization saved as {output_file}")

# 对每个 dataset 绘制图表
for dataset in datasets:
    plot_dataset(dataset)
