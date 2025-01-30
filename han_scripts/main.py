import torch
import torch.nn as nn
import torch.optim as optim
import flwr
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子以确保可复现性
torch.manual_seed(42)

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 构建一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # 输入层，10维输入
        self.fc2 = nn.Linear(50, 1)  # 输出层，输出一个标量

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建一些模拟数据（输入是10维，标签是1维）
X = torch.randn(100, 10)  # 100个样本，每个样本10维
y = torch.randn(100, 1)  # 100个标签，1维

# 将数据包装成DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 实例化模型并将其移动到 GPU（如果可用）
model = SimpleNN().to(device)

# 使用均方误差作为损失函数，Adam优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0

    for inputs, labels in train_loader:
        # 将数据移动到 GPU（如果可用）
        inputs, labels = inputs.to(device), labels.to(device)

        # 将梯度归零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 累计损失
        running_loss += loss.item()

    # 每个epoch结束后打印损失
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# 测试模型（使用相同的数据）
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 不计算梯度，节省内存
    test_input = torch.randn(5, 10).to(device)  # 测试输入5个样本并转移到GPU
    test_output = model(test_input)  # 模型预测
    print(f"Test input: {test_input}")
    print(f"Predicted output: {test_output}")
