import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(df.columns[j] + '(t-%d)' % i) for j in range(n_vars)]

    # 预测序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(df.columns[j] + '(t)') for j in range(n_vars)]
        else:
            names += [(df.columns[j] + '(t+%d)' % i) for j in range(n_vars)]

    # 把它们都放在一起
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # 删除含有NaN值的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 读取数据
data_path = 'data/cleaned_states_data.csv'
data = pd.read_csv(data_path)

# 标签编码“district”列
label_encoder = LabelEncoder()
data['district'] = label_encoder.fit_transform(data['district'])

# 提取月份作为周期性特征
data['month_sin'] = data['month'].apply(lambda x: np.sin(2 * np.pi * x / 12))
data['month_cos'] = data['month'].apply(lambda x: np.cos(2 * np.pi * x / 12))

# 特征和目标列
features_columns = ['district', 'temperature', 'population', 'month_sin', 'month_cos']
targets_columns = ['all sectors', 'commercial', 'industrial', 'residential']

# 初始化两个MinMaxScaler
features_scaler = MinMaxScaler()
targets_scaler = MinMaxScaler()

# 对特征和目标进行归一化处理
data[features_columns] = features_scaler.fit_transform(data[features_columns])
data[targets_columns] = targets_scaler.fit_transform(data[targets_columns])

# 将数据转换为监督学习格式
reframed = series_to_supervised(data[features_columns + targets_columns], 1, 1)

# 保存处理后的数据
processed_data_path = 'data/processed_states_data_supervised.csv'
reframed.to_csv(processed_data_path, index=False)

# 保存归一化模型
features_scaler_path = 'models/features_scaler.pkl'
targets_scaler_path = 'models/targets_scaler.pkl'
label_encoder_path = 'models/district_encoder.pkl'
joblib.dump(features_scaler, features_scaler_path)
joblib.dump(targets_scaler, targets_scaler_path)
joblib.dump(label_encoder, label_encoder_path)

print("特征工程完成，处理后的数据已保存至：", processed_data_path)
print("特征归一化模型已保存至：", features_scaler_path)
print("目标归一化模型已保存至：", targets_scaler_path)
print("地区编码器已保存至：", label_encoder_path)
