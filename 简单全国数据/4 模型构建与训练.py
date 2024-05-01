import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# 加载数据
train_data = pd.read_csv('data/train_data.csv')
features = train_data[['CPI', 'temperature', 'month_sin', 'month_cos']].values
targets = train_data[['all sectors', 'commercial', 'industrial']].values

# 重塑数据为LSTM需要的格式 [samples, time steps, features]
# 这里我们仍然使用一个时间步长，但是根据具体的时间序列分析需要，可以调整时间步长
features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

# 创建模型
model = Sequential()
model.add(LSTM(50, input_shape=(1, features.shape[2]), return_sequences=True))
model.add(LSTM(50))  # 可以考虑根据模型的复杂度需求增减LSTM层
model.add(Dense(3))  # 输出层有三个单元，对应三个目标变量

# 编译模型
model.compile(loss='mean_squared_error', optimizer=Adam())

# 显示模型概况
model.summary()

# 训练模型
history = model.fit(features, targets, epochs=100000, batch_size=32, validation_split=0.2, verbose=1)

# 保存模型
model_path = 'models/lstm_model.h5'
model.save(model_path)
print("训练模型已保存至：", model_path)
