import pandas as pd
from sklearn.model_selection import train_test_split

# 加载处理后的数据
data_path = 'data/processed_data.csv'
data = pd.read_csv(data_path, parse_dates=['date'])


# 方法1: 时间序列的顺序划分
def split_data_sequential(data, test_size=12):
    """
    按时间顺序将数据集的最后 `test_size` 个月的数据作为测试集。
    """
    training_data = data.iloc[:-test_size]
    testing_data = data.iloc[-test_size:]
    return training_data, testing_data


# 方法2: 随机划分
def split_data_random(data, test_size=0.2):
    """
    随机划分数据集为训练集和测试集，其中 `test_size` 表示测试集所占的比例。
    """
    training_data, testing_data = train_test_split(data, test_size=test_size, random_state=42)
    return training_data, testing_data


# 调用顺序划分函数
train_data, test_data = split_data_sequential(data)

# 调用随机划分方法，暂时注释
# train_data, test_data = split_data_random(data)

# 显示分割后的数据大小
print("训练集大小:", train_data.shape)
print("测试集大小:", test_data.shape)

# 保存训练集和测试集
train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)
