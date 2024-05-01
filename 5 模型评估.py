import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 加载测试数据
test_data = pd.read_csv('data/test_data.csv')

# 特征列和目标列
features_columns = ['district(t-1)', 'temperature(t-1)', 'population(t-1)', 'month_sin(t-1)', 'month_cos(t-1)']
target_columns = ['all sectors(t)', 'commercial(t)', 'industrial(t)', 'residential(t)']

# 提取特征和目标数据
test_features = test_data[features_columns].values
test_targets = test_data[target_columns].values

# 重塑特征数据为LSTM输入格式
test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))

# 加载模型
model_path = 'models/lstm_model.h5'
model = load_model(model_path)

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
    plt.title(f'Prediction vs Actual for {name}')
    plt.xlabel('Time Step')
    plt.ylabel('Electricity Consumption')
    plt.legend()
    plt.show()

print('模型评估完成。')
