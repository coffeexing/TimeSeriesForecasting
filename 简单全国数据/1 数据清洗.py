import pandas as pd

# 读取数据
data_path = 'data/raw.csv'
data = pd.read_csv(data_path)

# 显示数据的前几行以检查格式
print(data.head())

# 检查缺失值
print("检查缺失值：")
print(data.isnull().sum())

# 假设数据中无需填充缺失值，如果有缺失值可以根据需要填充或删除
data = data.fillna(method='ffill')  # 向前填充
data = data.dropna()  # 删除含有缺失值的行

# 删除不需要的列
data.drop('No', axis=1, inplace=True)

# 转换日期数据为Pandas datetime格式（如果还没有）
data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))

# 确保所有数据类型都正确
data['year'] = data['year'].astype(int)
data['month'] = data['month'].astype(int)
data['CPI'] = data['CPI'].astype(float)
data['temperature'] = data['temperature'].astype(float)
data['all sectors'] = data['all sectors'].astype(float)
data['commercial'] = data['commercial'].astype(float)
data['industrial'] = data['industrial'].astype(float)

# 导出清洗后的数据
clean_data_path = 'data/cleaned_data.csv'
data.to_csv(clean_data_path, index=False)

print("数据清洗完成，已保存至：", clean_data_path)
