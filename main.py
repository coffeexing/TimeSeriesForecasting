import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 从CSV文件中加载训练和测试数据
train_data = pd.read_csv('data/train_data.csv')
test_data = pd.read_csv('data/test_data.csv')

# 确保特征和目标列符合你的要求
features_columns = ['district(t-1)', 'temperature(t-1)', 'population(t-1)', 'month_sin(t-1)', 'month_cos(t-1)']
target_columns = ['all sectors(t)', 'commercial(t)', 'industrial(t)', 'residential(t)']

# 检查列名以确保正确使用
print("训练数据列名：", train_data.columns)
print("测试数据列名：", test_data.columns)

# 提取特征和目标列
features = train_data[features_columns].values
targets = train_data[target_columns].values

# 将特征重塑为LSTM输入格式 [样本数, 时间步数, 特征数]
features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

# 构建并编译LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(1, features.shape[2]), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(targets.shape[1]))

model.compile(loss='mean_squared_error', optimizer=Adam())

# 训练模型
history = model.fit(features, targets, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# 保存模型
model_path = 'models/lstm_model.h5'
model.save(model_path)
print("模型已保存至：", model_path)

# 测试数据集的特征和目标
test_features = test_data[features_columns].values
test_targets = test_data[target_columns].values

# 将特征重塑为LSTM输入格式
test_features = np.reshape(test_features, (test_features.shape[0], 1, test_features.shape[1]))

# 加载训练好的模型
model = load_model(model_path)

# 对测试数据进行预测
predictions = model.predict(test_features)

# 评估模型在测试数据上的性能
target_names = ['全部', '商业', '工业', '住宅']
metrics = {}

for i, name in enumerate(target_names):
    mse = mean_squared_error(test_targets[:, i], predictions[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_targets[:, i], predictions[:, i])
    r2 = r2_score(test_targets[:, i], predictions[:, i])
    metrics[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

    print(f'{name} - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}')

    plt.figure(figsize=(10, 5))
    plt.plot(test_targets[:, i], label='实际 ' + name)
    plt.plot(predictions[:, i], label='预测 ' + name)
    plt.title(f'{name} 的预测 vs 实际')
    plt.xlabel('时间步')
    plt.ylabel('用电量')
    plt.legend()
    plt.show()

print('模型评估完成。')
