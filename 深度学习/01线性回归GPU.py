import numpy as np
import torch
import torch.nn as nn

# ========== 设备配置（新增） ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 构造x：0~10，一共11个数字
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1) # 改成 (样本数,特征数)

# 构造y，真实公式 y = 2x + 1
y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

# 1. 定义线性回归模型类
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

# 2. 实例化模型
input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
model = model.to(device)  # ========== 模型放到GPU ==========

epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
    epoch += 1
    # 转tensor后移动到GPU
    inputs = torch.from_numpy(x_train).to(device)   # ========== tensor to GPU ==========
    labels = torch.from_numpy(y_train).to(device)   # ========== tensor to GPU ==========

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))

# ===== 训练完，查看 w 和 b =====
print('\n===== 训练结果 =====')
print('学到的权重 w =', model.linear.weight.item())
print('学到的偏置 b =', model.linear.bias.item())
print('真实公式: y = 2x + 1')

# 预测阶段：输入tensor也要放到GPU
with torch.no_grad():
    x_tensor = torch.from_numpy(x_train).to(device)
    predicted = model(x_tensor).cpu().numpy() # 计算完把结果搬回CPU再转numpy
print('预测值为：', predicted)

# 保存模型参数（不变）
torch.save(model.state_dict(), 'model.pkl')

# 加载模型：实例化模型，移动到对应device，再load权重
model_load = LinearRegressionModel(input_dim, output_dim)
model_load = model_load.to(device)
model_load.load_state_dict(torch.load('model.pkl'))
