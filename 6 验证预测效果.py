import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import load_model
import joblib

# 读取2023年数据
data_path = 'data/2023us_test.csv'
data_2023 = pd.read_csv(data_path)

# 加载保存的编码器和缩放器
district_encoder = joblib.load('models/district_encoder.pkl')
feature_scaler = joblib.load('models/features_scaler.pkl')
target_scaler = joblib.load('models/targets_scaler.pkl')

# 使用编码器对地区进行编码
data_2023['district'] = district_encoder.transform(data_2023['district'])

# 提取特征并添加周期性特征
data_2023['month_sin'] = np.sin(2 * np.pi * data_2023['month'] / 12)
data_2023['month_cos'] = np.cos(2 * np.pi * data_2023['month'] / 12)

# 提取并使用保存的缩放器对特征数据进行归一化
features_columns = ['district', 'temperature', 'population', 'month_sin', 'month_cos']
features_2023 = data_2023[features_columns].values

# 输出特征数据
print("特征数据（归一化前）：")
print(pd.DataFrame(features_2023, columns=features_columns))

scaled_features_2023 = feature_scaler.transform(features_2023)

# 输出归一化后的特征数据
print("特征数据（归一化后）：")
print(pd.DataFrame(scaled_features_2023, columns=features_columns))

# 重塑数据以适应LSTM输入格式
scaled_features_2023 = np.reshape(scaled_features_2023, (scaled_features_2023.shape[0], 1, scaled_features_2023.shape[1]))

# 加载训练好的模型
model = load_model('models/lstm_model.h5')

# 预测2023年用电量
predictions_2023 = model.predict(scaled_features_2023)

# 输出预测结果
print("预测结果（归一化）：")
print(predictions_2023)

# 反归一化预测结果
predicted_consumption_2023 = target_scaler.inverse_transform(predictions_2023)

# 输出反归一化后的预测结果
print("预测结果（反归一化）：")
print(predicted_consumption_2023)

# 从数据中提取实际值
actual_consumption_2023 = data_2023[['all sectors', 'commercial', 'industrial', 'residential']].values

# 绘制对比图
target_names = ['All Sectors', 'Commercial', 'Industrial', 'Residential']
months = range(1, 13)

for i, name in enumerate(target_names):
    plt.figure(figsize=(10, 5))
    plt.plot(months, actual_consumption_2023[:, i], label='Actual ' + name)
    plt.plot(months, predicted_consumption_2023[:, i], label='Predicted ' + name)
    plt.title(f'Actual vs Predicted for {name} in 2023')
    plt.xlabel('Month')
    plt.ylabel('Electricity Consumption')
    plt.legend()
    plt.show()
