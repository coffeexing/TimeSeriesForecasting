import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# 读取数据
data_path = 'data/states_time_data.csv'
data = pd.read_csv(data_path)

# 显示数据的前几行以检查格式
print(data.head())

# 检查缺失值
print("检查缺失值：")
print(data.isnull().sum())

# 将 'NM' 替换为 NaN
data.replace('NM', np.nan, inplace=True)

# 填充其他缺失值（如果有）
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# 转换日期数据为Pandas datetime格式（如果还没有）
data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
data.set_index('date', inplace=True)  # 设置日期为索引

# 使用 ARIMA 模型进行时间序列填充，特别是对 'residential' 列
# 注意：此函数现在只在发现序列中有 NaN 时执行 ARIMA 填充
def fill_with_arima(series, order):
    # 检查是否存在 NaN
    if series.isnull().any():
        # 替换 NaN 以适应 ARIMA 模型
        series_filled = series.fillna(method='ffill').fillna(method='bfill')
        model = ARIMA(series_filled, order=order)
        model_fit = model.fit()
        forecast = model_fit.predict(start=series.index[0], end=series.index[-1])
        # 只替换原先是 NaN 的部分
        series[series.isnull()] = forecast[series.isnull()]
    return series

# 应用 ARIMA 填充
data['residential'] = fill_with_arima(data['residential'], order=(1,1,1))

# 确保所有数据类型都正确
columns_to_convert = ['temperature', 'population', 'all sectors', 'commercial', 'industrial', 'residential']
for column in columns_to_convert:
    data[column] = data[column].astype(float)

# 导出清洗后的数据
clean_data_path = 'data/cleaned_states_data.csv'
data.to_csv(clean_data_path, index=False)

print("数据清洗完成，已保存至：", clean_data_path)
