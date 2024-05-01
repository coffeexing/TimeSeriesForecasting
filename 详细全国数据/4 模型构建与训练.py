import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

# 加载数据
train_data = pd.read_csv('data/train_data.csv')
features = train_data[['CPI', 'temperature', 'month_sin', 'month_cos', 'GDP', 'population', 'urbanization_rate', 'INDPRO']].values
targets = train_data[['all sectors', 'commercial', 'industrial', 'residential']].values

# 重塑数据为LSTM需要的格式 [samples, time steps, features]
features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

# 创建模型
model = Sequential()
model.add(LSTM(50, input_shape=(1, features.shape[2]), return_sequences=True))
model.add(LSTM(50))
model.add(Dense(4))  # 输出层调整为4，对应四个目标变量

# 编译模型
model.compile(loss='mean_squared_error', optimizer=Adam())

# 显示模型概况
model.summary()

# 训练模型
history = model.fit(features, targets, epochs=3000, batch_size=32, validation_split=0.2, verbose=1)  # 调整训练次数为实际需要的值

# 保存模型
model_path = 'models/lstm_model.h5'
model.save(model_path)
print("训练模型已保存至：", model_path)
