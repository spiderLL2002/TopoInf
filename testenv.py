import torch

print(torch.__version__)                # 查看pytorch安装的版本号
print(torch.cuda.device_count())        # 返回可以用的cuda（GPU）数量，0代表一个
print(torch.version.cuda)             # 查看cuda的版本
# 测试一个简单的 GPU 张量运算
'''print(torch.cuda.is_available())  一旦运行这一句，就会报错 '''     
x = torch.randn(10000, 10000).cuda()
y = torch.mm(x, x)
print(y)
print("Computation successful on GPU.")

from torch_geometric.data import Data

print("PyTorch version:", torch.__version__)
print("PyTorch Geometric installed successfully!")

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print("Graph data:", data)
