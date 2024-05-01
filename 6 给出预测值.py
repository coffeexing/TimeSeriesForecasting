import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# 加载模型和缩放器
model = load_model('models/lstm_model.h5')
feature_scaler = joblib.load('models/features_scaler.pkl')
target_scaler = joblib.load('models/targets_scaler.pkl')

# 准备特征数据（这里假设已经是预处理和序列化好的数据）
# 假设数据文件已经包含了2024年的预测所需的输入特征
test_data = pd.read_csv('data/test_data.csv')  # 修改为实际的文件名

# 重塑特征数据以适应LSTM输入格式
test_features = np.reshape(test_data.values, (test_data.shape[0], 1, test_data.shape[1]))

# 预测
predictions = model.predict(test_features)

# 反归一化预测结果
predictions = target_scaler.inverse_transform(predictions)

# 打印每个月的预测值和可视化预测结果
dates = pd.date_range(start='2024-01', periods=12, freq='M')
plt.figure(figsize=(14, 7))
target_names = ['All Sectors', 'Commercial', 'Industrial', 'Residential']

# 绘图
for i, label in enumerate(target_names):
    plt.plot(dates, predictions[:, i], marker='o', label=label)
    for idx, val in enumerate(predictions[:, i]):
        plt.text(dates[idx], val, f'{val:.2f}')

plt.xlabel('Month in 2024')
plt.ylabel('Electricity Consumption')
plt.title('Monthly Electricity Consumption Prediction for 2024')
plt.legend()
plt.grid(True)
plt.show()

# 可视化为柱状图
plt.figure(figsize=(14, 7))
width = 0.20  # 柱状图的宽度
months = np.arange(len(dates))  # 月份标签

for i, label in enumerate(target_names):
    plt.bar(months - 1.5*width + i*width, predictions[:, i], width=width, label=label)

plt.xlabel('Month in 2024')
plt.ylabel('Electricity Consumption')
plt.title('Monthly Electricity Consumption Prediction for 2024')
plt.xticks(months, [date.strftime('%Y-%m') for date in dates], rotation=45)
plt.legend()
plt.show()
