import pandas as pd

# 加载处理后的数据
data_path = 'data/processed_states_data_supervised.csv'
data = pd.read_csv(data_path)

# 顺序划分
def split_data_sequential(data, test_size=12*48):
    """
    按时间顺序将数据集的最后 `test_size` 个月的数据作为测试集。
    """
    training_data = data.iloc[:-test_size]
    testing_data = data.iloc[-test_size:]
    return training_data, testing_data

# 调用顺序划分函数
train_data, test_data = split_data_sequential(data)

# 显示分割后的数据大小
print("训练集大小:", train_data.shape)
print("测试集大小:", test_data.shape)

# 保存训练集和测试集
train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

print("训练集和测试集已保存。")
