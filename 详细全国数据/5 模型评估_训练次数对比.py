import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载测试数据
test_data = pd.read_csv('data/processed_data.csv').tail(12)

# 加载特征和目标的Scaler对象
feature_scaler = joblib.load('models/features_scaler.pkl')
target_scaler = joblib.load('models/targets_scaler.pkl')

# 添加周期性特征
test_data['month_sin'] = np.sin(2 * np.pi * test_data['month'] / 12)
test_data['month_cos'] = np.cos(2 * np.pi * test_data['month'] / 12)

# 添加其他特征
features_columns = ['CPI', 'temperature', 'month_sin', 'month_cos', 'GDP', 'population', 'urbanization_rate', 'INDPRO']
targets_columns = ['all sectors', 'commercial', 'industrial', 'residential']
test_features = test_data[features_columns].values
test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))
test_targets = target_scaler.inverse_transform(test_data[targets_columns].values)

# 模型和绘图
epochs = [100, 500, 1500, 2500, 5000]
metrics = {'MSE': mean_squared_error, 'RMSE': lambda y, p: np.sqrt(mean_squared_error(y, p)),
           'MAE': mean_absolute_error, 'R²': r2_score}
results = {e: {m: [] for m in metrics} for e in epochs}

# 生成用电量预测图
for j, name in enumerate(targets_columns):
    plt.figure(figsize=(10, 5))
    for i, epoch in enumerate(epochs):
        model = load_model(f'models/{epoch}_lstm_model.h5')
        predictions = model.predict(test_features)
        predictions_inversed = target_scaler.inverse_transform(predictions)

        plt.plot(test_targets[:, j], label='Actual Consumption' if i == 0 else "")
        plt.plot(predictions_inversed[:, j], label=f'{epoch} Epochs')

    plt.title(f'各行业不同模型的预测值和真实值: {name}')
    plt.xlabel('Time Step')
    plt.ylabel('Electricity Consumption')
    plt.legend()
    plt.show()

# 指标对比图
for metric_name in metrics:
    plt.figure(figsize=(10, 5))
    for target_idx, target_name in enumerate(targets_columns):
        values = [results[epoch][metric_name][target_idx] for epoch in epochs]
        plt.plot(epochs, values, marker='o', label=target_name)

    plt.title(f'{metric_name} Comparison Across Targets')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
