import numpy as np

# 构造x：0~10，一共11个数字
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1) # 改成 (样本数,特征数)
x_train.shape # 输出 (11,1)

# 构造y，真实公式 y = 2x + 1
y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
y_train.shape # 输出 (11,1)

import torch
import torch.nn as nn

# 1. 定义线性回归模型类
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # 全连接层：input_dim个输入特征，output_dim个输出
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 前向传播：数据x送入linear层，直接输出，不加激活函数
        out = self.linear(x)
        return out

# 2. 实例化模型：单变量线性回归（输入1维，输出1维）
input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

for epoch in range(epochs):
    epoch += 1
    # 注意转行成tensor
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)

    # 梯度要清零每一次迭代
    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, labels)

    # 返向传播
    loss.backward()

    # 更新权重参数
    optimizer.step()

    if epoch % 50 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))



# ===== 训练完，查看 w 和 b =====
print('\n===== 训练结果 =====')
print('学到的权重 w =', model.linear.weight.item())
print('学到的偏置 b =', model.linear.bias.item())
print('真实公式: y = 2x + 1')


with torch.no_grad():
    predicted = model(torch.from_numpy(x_train)).numpy()
print('预测值为：', predicted)


# 保存：只保存模型参数（state_dict，推荐，文件体积更小、更安全）
torch.save(model.state_dict(), 'model.pkl')
# 读取：先实例化和原来结构完全一样的model，再加载权重
model.load_state_dict(torch.load('model.pkl'))
