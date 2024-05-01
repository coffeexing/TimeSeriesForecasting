import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# 读取数据
data_path = 'data/cleaned_data.csv'
data = pd.read_csv(data_path)

# 提取月份作为周期性特征
data['month_sin'] = data['month'].apply(lambda x: np.sin(2 * np.pi * x / 12))
data['month_cos'] = data['month'].apply(lambda x: np.cos(2 * np.pi * x / 12))

# 特征和目标列
features_columns = ['CPI', 'temperature', 'GDP', 'population', 'urbanization_rate', 'INDPRO', 'month_sin', 'month_cos']
targets_columns = ['all sectors', 'commercial', 'industrial', 'residential']

# 初始化两个MinMaxScaler
features_scaler = MinMaxScaler()
targets_scaler = MinMaxScaler()

# 对特征和目标进行归一化处理
data[features_columns] = features_scaler.fit_transform(data[features_columns])
data[targets_columns] = targets_scaler.fit_transform(data[targets_columns])

# 保存处理后的数据
processed_data_path = 'data/processed_data.csv'
data.to_csv(processed_data_path, index=False)

# 保存归一化模型
features_scaler_path = 'models/features_scaler.pkl'
targets_scaler_path = 'models/targets_scaler.pkl'
joblib.dump(features_scaler, features_scaler_path)
joblib.dump(targets_scaler, targets_scaler_path)

print("特征工程完成，处理后的数据已保存至：", processed_data_path)
print("特征归一化模型已保存至：", features_scaler_path)
print("目标归一化模型已保存至：", targets_scaler_path)
