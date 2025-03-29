import os
import json
import pandas as pd
import glob


def get_cache_dir():
  """获取缓存目录"""
  return os.path.join(os.getcwd(), 'cache')


def load_column_names():
  """加载数据文件的列名"""
  try:
    # 仅读取CSV文件的第一行来获取列名
    df_header = pd.read_csv('data/merged_data.csv', nrows=0)
    column_names = df_header.columns.tolist()
    return column_names
  except Exception as e:
    print(f"读取列名时出错: {e}")
    return []


def load_all_cached_columns():
  """加载所有列缓存文件"""
  cache_dir = get_cache_dir()
  column_files = glob.glob(os.path.join(cache_dir, 'column_*.json'))

  # 检查是否有缓存文件
  if not column_files:
    print("找不到缓存文件！")
    return {}

  # 读取所有列缓存
  all_columns_data = {}
  for file_path in column_files:
    try:
      with open(file_path, 'r', encoding='utf-8') as f:
        column_data = json.load(f)
        file_name = os.path.basename(file_path)
        column_key = file_name.replace('column_', '').replace('.json', '')
        all_columns_data[column_key] = column_data
    except Exception as e:
      print(f"读取文件 {file_path} 出错: {e}")

  return all_columns_data


def extract_causal_relationships(all_columns_data, column_names):
  """提取所有因果关系和延迟"""
  causal_relationships = []

  for column_key, data in all_columns_data.items():
    # 检查是否有因果关系数据
    if 'causeswithdelay' in data:
      for relation_str, delay in data['causeswithdelay'].items():
        # 将字符串形式的元组解析为实际的索引
        try:
          # 处理元组字符串 "(a, b)"
          relation = eval(relation_str)
          effect_idx, cause_idx = relation

          # 获取列名
          effect_name = column_names[effect_idx] if effect_idx < len(
              column_names
          ) else f"未知列 {effect_idx}"
          cause_name = column_names[cause_idx] if cause_idx < len(
              column_names
          ) else f"未知列 {cause_idx}"

          causal_relationships.append(
              {
                  'effect_idx': effect_idx,
                  'cause_idx': cause_idx,
                  'effect_name': effect_name,
                  'cause_name': cause_name,
                  'delay': delay,
                  'column_key': column_key
              }
          )
        except Exception as e:
          print(f"解析关系 {relation_str} 出错: {e}")

  return causal_relationships


def main():
  """主函数"""
  print("正在从缓存中提取因果关系...")

  # 加载列名
  column_names = load_column_names()
  if not column_names:
    print("警告: 无法加载列名，将使用索引号代替。")
  else:
    print(f"成功加载 {len(column_names)} 个列名")

  # 加载缓存数据
  all_columns_data = load_all_cached_columns()
  if not all_columns_data:
    print("没有找到缓存数据，无法提取因果关系。")
    return

  print(f"成功读取 {len(all_columns_data)} 个缓存文件")

  # 提取所有因果关系
  causal_relationships = extract_causal_relationships(
      all_columns_data, column_names
  )

  if not causal_relationships:
    print("未找到任何因果关系。")
    return

  print(f"提取到 {len(causal_relationships)} 个因果关系")

  # 筛选延迟小于1000的关系
  filtered_relationships = [
      r for r in causal_relationships if r['delay'] < 1000
  ]

  print(f"延迟小于1000的因果关系共有 {len(filtered_relationships)} 个")

  # 将结果保存到CSV文件
  output_df = pd.DataFrame(filtered_relationships)
  output_path = os.path.join('output', 'causal_delays_less_than_1000.csv')
  output_df.to_csv(output_path, index=False)

  print(f"结果已保存至: {output_path}")

  # 显示部分结果
  if filtered_relationships:
    print("\n延迟小于1000的因果关系（前10个）:")
    for i, rel in enumerate(filtered_relationships[:10]):
      print(
          f"{i+1}. \"{rel['cause_name']}\" 影响 \"{rel['effect_name']}\"，延迟: {rel['delay']} 时间步"
      )


if __name__ == "__main__":
  main()
