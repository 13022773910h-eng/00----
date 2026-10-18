import torch
# 0维 标量
scalar = torch.tensor(2.5)
print(scalar.shape) # torch.Size([])

# 1维 向量
vec = torch.tensor([1,2,3])
print(vec.shape) # torch.Size([3])

# 2维 矩阵
mat = torch.tensor([[1,2],[3,4]])
print(mat.shape) # torch.Size([2, 2])

# 3维 张量
tensor3d = torch.rand(2,3,4)
print(tensor3d.shape) # torch.Size([2, 3, 4])
