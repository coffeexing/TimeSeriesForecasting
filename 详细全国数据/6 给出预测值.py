import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# 加载测试数据
test_data = pd.read_csv('data/processed_data.csv').tail(12)

# 加载特征和目标的Scaler对象
feature_scaler = joblib.load('models/features_scaler.pkl')
target_scaler = joblib.load('models/targets_scaler.pkl')

# 添加完整的特征和目标列
feature_columns = ['CPI', 'temperature', 'month_sin', 'month_cos', 'GDP', 'population', 'urbanization_rate', 'INDPRO']
target_columns = ['all sectors', 'commercial', 'industrial', 'residential']
test_features = test_data[feature_columns].values
test_targets = test_data[target_columns].values

# 重塑特征数据以适应LSTM输入
test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))

# 加载模型
model = load_model('models/lstm_model.h5')

# 预测
predictions = model.predict(test_features)

# 反归一化预测结果
predictions = target_scaler.inverse_transform(predictions)

# 打印每个月的预测值
dates = pd.date_range(start='2024-01', periods=12, freq='M')
for i, date in enumerate(dates):
    print(f"{date.strftime('%Y-%m')}: All Sectors={predictions[i, 0]:.2f}, Commercial={predictions[i, 1]:.2f}, Industrial={predictions[i, 2]:.2f}, Residential={predictions[i, 3]:.2f}")

# 可视化预测结果
plt.figure(figsize=(14, 7))
target_names = ['All Sectors', 'Commercial', 'Industrial', 'Residential']
for i, label in enumerate(target_names):
    plt.plot(dates, predictions[:, i], marker='o', label=label)

plt.xlabel('Month in 2024')
plt.ylabel('Electricity Consumption')
plt.title('Monthly Electricity Consumption Prediction for 2024')
plt.legend()
plt.show()

# 显示柱状图
plt.figure(figsize=(14, 7))
width = 0.20  # 柱状图的宽度
months = np.arange(len(dates))  # 月份标签

# 为每个目标绘制柱状图
bars1 = plt.bar(months - 1.5*width, predictions[:, 0], width=width, label='All Sectors')
bars2 = plt.bar(months - 0.5*width, predictions[:, 1], width=width, label='Commercial')
bars3 = plt.bar(months + 0.5*width, predictions[:, 2], width=width, label='Industrial')
bars4 = plt.bar(months + 1.5*width, predictions[:, 3], width=width, label='Residential')

# 在柱状图上添加数值标签
for bars in (bars1, bars2, bars3, bars4):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Month in 2024')
plt.ylabel('Electricity Consumption')
plt.title('Monthly Electricity Consumption Prediction for 2024')
plt.xticks(months, [date.strftime('%Y-%m') for date in dates])
plt.legend()
plt.show()
