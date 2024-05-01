import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载处理后的测试数据
test_data = pd.read_csv('data/test_data.csv')

# 加载归一化参数
features_scaler = joblib.load('models/features_scaler.pkl')
targets_scaler = joblib.load('models/targets_scaler.pkl')

# 准备特征和目标列
features_columns = [col for col in test_data.columns if '(t-1)' in col]
targets_columns = [col for col in test_data.columns if '(t)' in col]

# 获取特征和目标数据
test_features = test_data[features_columns].values
test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))
test_targets = test_data[targets_columns].values

# 存储不同训练次数的模型预测结果
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
        predictions_inversed = targets_scaler.inverse_transform(predictions)

        plt.plot(test_targets[:, j], label='Actual Consumption' if i == 0 else "")
        plt.plot(predictions_inversed[:, j], label=f'{epoch} Epochs')

        # 计算指标
        for metric_name, metric in metrics.items():
            results[epoch][metric_name].append(metric(test_targets[:, j], predictions_inversed[:, j]))

    plt.title(f'Predictions and Actuals for Different Epochs: {name}')
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

    plt.title(f'{metric_name} Comparison Across Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
