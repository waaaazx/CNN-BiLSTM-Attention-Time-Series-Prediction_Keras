import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# 超参数
TIME_STEPS = 20
INPUT_DIMS = 7
lstm_units = 64
BATCH_SIZE = 64
EPOCHS = 10

# 自定义注意力机制模块
class AttentionBlock(nn.Module):
    def __init__(self, input_dim, single_attention_vector=False):
        super(AttentionBlock, self).__init__()
        self.single_attention_vector = single_attention_vector
        self.dense = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, input_dim)
        original_size = x.size()
        
        # 计算注意力权重
        a = self.dense(x)  # (batch_size, time_steps, input_dim)
        a = self.softmax(a)
        
        if self.single_attention_vector:
            a = torch.mean(a, dim=1, keepdim=True)  # (batch_size, 1, input_dim)
            a = a.repeat(1, original_size[1], 1)    # (batch_size, time_steps, input_dim)
        
        # 应用注意力权重
        output = x * a
        return output

# 定义完整的模型
class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        
        self.conv1d = nn.Conv1d(INPUT_DIMS, 64, kernel_size=1)
        self.dropout1 = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(64, lstm_units, bidirectional=True, batch_first=True)
        self.attention = AttentionBlock(lstm_units * 2)  # 双向LSTM输出维度翻倍
        self.dropout2 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(TIME_STEPS * lstm_units * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 输入x形状: (batch_size, time_steps, input_dims)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_dims, time_steps) 用于Conv1d
        
        x = self.conv1d(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        x = x.permute(0, 2, 1)  # 转换回 (batch_size, time_steps, features)
        
        # LSTM处理
        x, _ = self.bilstm(x)
        x = self.dropout2(x)
        
        # 注意力机制
        x = self.attention(x)
        
        # 输出层
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

# 创建数据集函数
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back), :])
        dataY.append(dataset[i + look_back, 0])  # 只预测第一列（pollution）
    return np.array(dataX), np.array(dataY)

# 自定义Dataset类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 加载数据
data = pd.read_csv("./pollution.csv")
data = data.drop(['date', 'wnd_dir'], axis=1)
print(data.columns)
print(data.shape)

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 创建训练数据集
pollution_data = scaled_data[:, 0].reshape(-1, 1)
train_X, train_Y = create_dataset(scaled_data, TIME_STEPS)
print(f"Train shapes: X={train_X.shape}, Y={train_Y.shape}")

# 创建数据加载器
dataset = TimeSeriesDataset(train_X, train_Y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttentionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}')

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': scaler
}, "pollution_model.pth")
