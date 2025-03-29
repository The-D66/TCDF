import pandas as pd

# 读取CSV文件
input_file = 'data/merged_data.csv'
output_file = 'data/merged_data_no_first.csv'

# 读取CSV文件，使用UTF-8编码
df = pd.read_csv(input_file, encoding='utf-8')

# 删除第一列
df = df.iloc[:, 1:]

# 保存处理后的文件，使用UTF-8编码
df.to_csv(output_file, index=False, encoding='utf-8')

print(f'处理完成！\n原始文件：{input_file}\n新文件：{output_file}')
