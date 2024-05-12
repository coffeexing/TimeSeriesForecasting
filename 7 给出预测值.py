import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
import joblib

# 读取2024年的数据
data_path = 'data/2024.csv'
data_2024 = pd.read_csv(data_path)

# 加载已保存的缩放器和编码器
district_encoder = joblib.load('models/district_encoder.pkl')
feature_scaler = joblib.load('models/features_scaler.pkl')
target_scaler = joblib.load('models/targets_scaler.pkl')

# 使用编码器对地区进行编码
data_2024['district'] = district_encoder.transform(data_2024['district'])

# 提取特征并添加周期性特征
data_2024['month_sin'] = np.sin(2 * np.pi * data_2024['month'] / 12)
data_2024['month_cos'] = np.cos(2 * np.pi * data_2024['month'] / 12)

# 提取特征数据
features_columns = ['district', 'temperature', 'population', 'month_sin', 'month_cos']
features_2024 = data_2024[features_columns].values

# 归一化特征数据
scaled_features_2024 = feature_scaler.transform(features_2024)

# 重塑数据以适应LSTM输入格式
scaled_features_2024 = np.reshape(scaled_features_2024, (scaled_features_2024.shape[0], 1, scaled_features_2024.shape[1]))

# 加载训练好的模型
model = load_model('models/lstm_model.h5')

# 预测2024年的用电量
predictions_2024 = model.predict(scaled_features_2024)

# 反归一化预测结果
predicted_consumption_2024 = target_scaler.inverse_transform(predictions_2024)

# 将预测结果存入数据框并输出
predictions_df = pd.DataFrame(predicted_consumption_2024, columns=['All Sectors', 'Commercial', 'Industrial', 'Residential'])
print("Predicted electricity consumption for 2024:")
print(predictions_df)

# 绘制预测结果（折线图）
months = pd.date_range(start='2024-01', periods=12, freq='M')

plt.figure(figsize=(12, 6))
for col in predictions_df.columns:
    plt.plot(months, predictions_df[col], marker='o', label=col)

plt.title('Predicted Electricity Consumption by Sector for 2024')
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.legend()
plt.grid(True)
plt.xticks(months, [month.strftime('%b') for month in months])
plt.show()

# 绘制预测结果（柱状图）
plt.figure(figsize=(14, 7))
width = 0.20  # 柱状图的宽度
months_idx = np.arange(len(months))  # 月份标签

# 为每个目标绘制柱状图
bars1 = plt.bar(months_idx - 1.5*width, predictions_df['All Sectors'], width=width, label='All Sectors')
bars2 = plt.bar(months_idx - 0.5*width, predictions_df['Commercial'], width=width, label='Commercial')
bars3 = plt.bar(months_idx + 0.5*width, predictions_df['Industrial'], width=width, label='Industrial')
bars4 = plt.bar(months_idx + 1.5*width, predictions_df['Residential'], width=width, label='Residential')

# 在柱状图上添加数值标签
for bars in (bars1, bars2, bars3, bars4):
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Predicted Electricity Consumption by Sector for 2024 (Bar Chart)')
plt.xticks(months_idx, [month.strftime('%b') for month in months])
plt.legend()
plt.show()
