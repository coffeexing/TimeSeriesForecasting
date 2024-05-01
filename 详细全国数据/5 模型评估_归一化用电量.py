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

# 添加其他特征
features_columns = ['CPI', 'temperature', 'month_sin', 'month_cos', 'GDP', 'population', 'urbanization_rate', 'INDPRO']
targets_columns = ['all sectors', 'commercial', 'industrial', 'residential']

# 选择正确的特征和目标
test_features = test_data[features_columns].values
test_targets = test_data[targets_columns].values

# 重塑特征数据以适应LSTM输入
test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))

# 加载模型
model = load_model('models/lstm_model.h5')

# 预测
predictions = model.predict(test_features)

# 可视化和计算每个目标的评估指标
target_names = ['All Sectors', 'Commercial', 'Industrial', 'Residential']
metrics = {}

for i, name in enumerate(target_names):
    mse = mean_squared_error(test_targets[:, i], predictions[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_targets[:, i], predictions[:, i])
    r2 = r2_score(test_targets[:, i], predictions[:, i])
    metrics[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

    # 打印指标
    print(f'{name} - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}')

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(test_targets[:, i], label='Actual ' + name)
    plt.plot(predictions[:, i], label='Predicted ' + name)
    plt.title('Prediction vs Actual for ' + name)
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Consumption')
    plt.legend()
    plt.show()

print('模型评估完毕！')
