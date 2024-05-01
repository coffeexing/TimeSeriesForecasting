import pandas as pd

# 读取数据
data_path = 'data/states_data.csv'
data = pd.read_csv(data_path)

# 显示数据的前几行以检查格式
print(data.head())

# 检查缺失值
print("检查缺失值：")
print(data.isnull().sum())

# 填充缺失值
data = data.fillna(method='ffill')  # 向前填充
data = data.dropna()  # 删除含有缺失值的行

# 删除不需要的列
data.drop('No', axis=1, inplace=True)

# 转换日期数据为Pandas datetime格式（如果还没有）
data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))

# 确保所有数据类型都正确
columns_to_convert = ['CPI', 'temperature', 'GDP', 'population', 'urbanization_rate', 'INDPRO',
                      'all sectors', 'commercial', 'industrial', 'residential']
for column in columns_to_convert:
    data[column] = data[column].astype(float)

# 导出清洗后的数据
clean_data_path = 'data/cleaned_data.csv'
data.to_csv(clean_data_path, index=False)

print("数据清洗完成，已保存至：", clean_data_path)
