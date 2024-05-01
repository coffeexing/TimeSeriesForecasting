import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 加载测试数据
test_data = pd.read_csv('data/test_data.csv')

# 添加周期性特征
test_data['month_sin'] = np.sin(2 * np.pi * test_data['month'] / 12)
test_data['month_cos'] = np.cos(2 * np.pi * test_data['month'] / 12)

# 选择正确的特征
test_features = test_data[['CPI', 'temperature', 'month_sin', 'month_cos']].values
test_targets = test_data[['all sectors', 'commercial', 'industrial']].values

# 重塑特征数据以适应LSTM输入
test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))

# 加载模型
model = load_model('models/lstm_model.h5')

# 预测
predictions = model.predict(test_features)

# 计算MSE, RMSE, MAE, R²
mse = mean_squared_error(test_targets, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_targets, predictions)
r2 = r2_score(test_targets, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')

# 可视化预测结果与实际值对比
target_names = ['All Sectors', 'Commercial', 'Industrial']
for i, name in enumerate(target_names):
    plt.figure(figsize=(10, 5))
    plt.plot(test_targets[:, i], label='Actual ' + name)
    plt.plot(predictions[:, i], label='Predicted ' + name)
    plt.title('Prediction vs Actual for ' + name)
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Consumption')
    plt.legend()
    plt.show()

print('模型评估完毕！')
