import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# 加载训练数据
train_data = pd.read_csv('data/train_data.csv')

# 特征列和目标列
features_columns = ['district(t-1)', 'temperature(t-1)', 'population(t-1)', 'month_sin(t-1)', 'month_cos(t-1)']
target_columns = ['all sectors(t)', 'commercial(t)', 'industrial(t)', 'residential(t)']

# 提取特征和目标数据
features = train_data[features_columns].values
targets = train_data[target_columns].values

# 重塑特征数据为LSTM输入格式 [样本数, 时间步数, 特征数]
features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

# 创建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(1, features.shape[2]), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(targets.shape[1]))

# 编译模型
model.compile(loss='mean_squared_error', optimizer=Adam())

# 显示模型概况
model.summary()

# 训练模型
history = model.fit(features, targets, epochs=500, batch_size=32, validation_split=0.2, verbose=1)

# 保存模型
model_path = 'models/lstm_model.h5'
model.save(model_path)
print("训练模型已保存至：", model_path)
