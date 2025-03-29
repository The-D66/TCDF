import TCDF
import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
import pylab
import copy
import matplotlib.pyplot as plt
import os
import sys
import json
import hashlib
from joblib import Parallel, delayed
import multiprocessing

# os.chdir(os.path.dirname(sys.argv[0])) #uncomment this line to run in VSCode


def get_cache_dir():
  """获取缓存目录，如果不存在则创建"""
  cache_dir = 'cache'
  if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
  return cache_dir


def get_dataset_cache_key(datafile, args):
  """生成数据集的缓存键，基于数据文件和关键参数"""
  # 读取数据文件的前几行来生成哈希
  df = pd.read_csv(datafile, nrows=1000)
  data_hash = hashlib.md5(df.to_string().encode()).hexdigest()

  # 组合关键参数
  key_params = {
      'data_hash': data_hash,
      'epochs': args.epochs,
      'kernel_size': args.kernel_size,
      'hidden_layers': args.hidden_layers,
      'learning_rate': args.learning_rate,
      'optimizer': args.optimizer,
      'dilation_coefficient': args.dilation_coefficient,
      'significance': args.significance
  }

  return hashlib.md5(json.dumps(key_params,
                                sort_keys=True).encode()).hexdigest()


def get_column_cache_key(dataset_key, column_idx, column_name):
  """生成单个列的缓存键"""
  key_params = {
      'dataset_key': dataset_key,
      'column_idx': column_idx,
      'column_name': column_name
  }
  return hashlib.md5(json.dumps(key_params,
                                sort_keys=True).encode()).hexdigest()


def load_column_cache(dataset_key, column_idx, column_name):
  """从缓存加载单个列的结果"""
  cache_dir = get_cache_dir()
  column_key = get_column_cache_key(dataset_key, column_idx, column_name)
  cache_file = os.path.join(cache_dir, f'column_{column_key}.json')

  if os.path.exists(cache_file):
    with open(cache_file, 'r', encoding='utf-8') as f:
      return json.load(f)
  return None


def save_column_cache(
    dataset_key, column_idx, column_name, causes, causeswithdelay, realloss,
    scores
):
  """保存单个列的结果到缓存"""
  cache_dir = get_cache_dir()
  column_key = get_column_cache_key(dataset_key, column_idx, column_name)
  cache_file = os.path.join(cache_dir, f'column_{column_key}.json')

  column_result = {
      'causes': causes,
      'causeswithdelay': {
          str(k): v
          for k, v in causeswithdelay.items()
      },  # 将元组键转换为字符串
      'realloss': realloss,
      'scores': scores
  }

  with open(cache_file, 'w', encoding='utf-8') as f:
    json.dump(column_result, f, ensure_ascii=False, indent=2)

  # 更新列缓存索引
  index_file = os.path.join(cache_dir, f'index_{dataset_key}.json')
  if os.path.exists(index_file):
    with open(index_file, 'r', encoding='utf-8') as f:
      index = json.load(f)
  else:
    index = {'columns': {}}

  index['columns'][str(column_idx)] = {
      'name': column_name,
      'cache_key': column_key
  }

  with open(index_file, 'w', encoding='utf-8') as f:
    json.dump(index, f, ensure_ascii=False, indent=2)


def load_dataset_cache(dataset_key):
  """加载整个数据集的缓存索引"""
  cache_dir = get_cache_dir()
  index_file = os.path.join(cache_dir, f'index_{dataset_key}.json')

  if os.path.exists(index_file):
    with open(index_file, 'r', encoding='utf-8') as f:
      return json.load(f)
  return None


def check_positive(value):
  """Checks if argument is positive integer (larger than zero)."""
  ivalue = int(value)
  if ivalue <= 0:
    raise argparse.ArgumentTypeError("%s should be positive" % value)
  return ivalue


def check_zero_or_positive(value):
  """Checks if argument is positive integer (larger than or equal to zero)."""
  ivalue = int(value)
  if ivalue < 0:
    raise argparse.ArgumentTypeError("%s should be positive" % value)
  return ivalue


class StoreDictKeyPair(argparse.Action):
  """Creates dictionary containing datasets as keys and ground truth files as values."""
  def __call__(self, parser, namespace, values, option_string=None):
    my_dict = {}
    for kv in values.split(","):
      k, v = kv.split("=")
      my_dict[k] = v
    setattr(namespace, self.dest, my_dict)


def getextendeddelays(gtfile, columns):
  """Collects the total delay of indirect causal relationships."""
  gtdata = pd.read_csv(gtfile, header=None)

  readgt = dict()
  effects = gtdata[1]
  causes = gtdata[0]
  delays = gtdata[2]
  gtnrrelations = 0
  pairdelays = dict()
  for k in range(len(columns)):
    readgt[k] = []
  for i, (key, value) in enumerate(zip(effects, causes)):
    readgt[key].append(value)
    pairdelays[(key, value)] = delays[i]
    gtnrrelations += 1

  g = nx.DiGraph()
  g.add_nodes_from(readgt.keys())
  for e in readgt:
    cs = readgt[e]
    for c in cs:
      g.add_edge(c, e)

  extendedreadgt = copy.deepcopy(readgt)

  for c1 in range(len(columns)):
    for c2 in range(len(columns)):
      paths = list(
          nx.all_simple_paths(g, c1, c2, cutoff=2)
      )  #indirect path max length 3, no cycles

      if len(paths) > 0:
        for path in paths:
          for p in path[:-1]:
            if p not in extendedreadgt[path[-1]]:
              extendedreadgt[path[-1]].append(p)

  extendedgtdelays = dict()
  for effect in extendedreadgt:
    causes = extendedreadgt[effect]
    for cause in causes:
      if (effect, cause) in pairdelays:
        delay = pairdelays[(effect, cause)]
        extendedgtdelays[(effect, cause)] = [delay]
      else:
        #find extended delay
        paths = list(
            nx.all_simple_paths(g, cause, effect, cutoff=2)
        )  #indirect path max length 3, no cycles
        extendedgtdelays[(effect, cause)] = []
        for p in paths:
          delay = 0
          for i in range(len(p) - 1):
            delay += pairdelays[(p[i + 1], p[i])]
          extendedgtdelays[(effect, cause)].append(delay)

  return extendedgtdelays, readgt, extendedreadgt


def evaluate(gtfile, validatedcauses, columns):
  """Evaluates the results of TCDF by comparing it to the ground truth graph, and calculating precision, recall and F1-score. F1'-score, precision' and recall' include indirect causal relationships."""
  extendedgtdelays, readgt, extendedreadgt = getextendeddelays(gtfile, columns)
  FP = 0
  FPdirect = 0
  TPdirect = 0
  TP = 0
  FN = 0
  FPs = []
  FPsdirect = []
  TPsdirect = []
  TPs = []
  FNs = []
  for key, value in readgt.items():
    for v in validatedcauses[key]:
      if v not in extendedreadgt[key]:
        FP += 1
        FPs.append((key, v))
      else:
        TP += 1
        TPs.append((key, v))
      if v not in readgt[key]:
        FPdirect += 1
        FPsdirect.append((key, v))
      else:
        TPdirect += 1
        TPsdirect.append((key, v))
    for v in readgt[key]:
      if v not in validatedcauses[key]:
        FN += 1
        FNs.append((key, v))

  print("Total False Positives': ", FP)
  print("Total True Positives': ", TP)
  print("Total False Negatives: ", FN)
  print("Total Direct False Positives: ", FPdirect)
  print("Total Direct True Positives: ", TPdirect)
  print("TPs': ", TPs)
  print("FPs': ", FPs)
  print("TPs direct: ", TPsdirect)
  print("FPs direct: ", FPsdirect)
  print("FNs: ", FNs)
  precision = recall = 0.

  if float(TP + FP) > 0:
    precision = TP / float(TP + FP)
  print("Precision': ", precision)
  if float(TP + FN) > 0:
    recall = TP / float(TP + FN)
  print("Recall': ", recall)
  if (precision + recall) > 0:
    F1 = 2 * (precision * recall) / (precision + recall)
  else:
    F1 = 0.
  print(
      "F1' score: ", F1, "(includes direct and indirect causal relationships)"
  )

  precision = recall = 0.
  if float(TPdirect + FPdirect) > 0:
    precision = TPdirect / float(TPdirect + FPdirect)
  print("Precision: ", precision)
  if float(TPdirect + FN) > 0:
    recall = TPdirect / float(TPdirect + FN)
  print("Recall: ", recall)
  if (precision + recall) > 0:
    F1direct = 2 * (precision * recall) / (precision + recall)
  else:
    F1direct = 0.
  print("F1 score: ", F1direct, "(includes only direct causal relationships)")
  return FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct


def evaluatedelay(extendedgtdelays, alldelays, TPs, receptivefield):
  """Evaluates the delay discovery of TCDF by comparing the discovered time delays with the ground truth."""
  zeros = 0
  total = 0.
  for i, tp in enumerate(TPs):
    discovereddelay = alldelays[tp]
    gtdelays = extendedgtdelays[tp]
    for d in gtdelays:
      if d <= receptivefield:
        total += 1.
        error = d - discovereddelay
        if error == 0:
          zeros += 1

      else:
        next

  if zeros == 0:
    return 0.
  else:
    return zeros / float(total)


# 为归一化处理添加并行函数
def zscore_normalize_column(df, column):
  """并行处理Z-score归一化单列"""
  series = df[column]
  mean = series.mean()
  std = series.std()

  if std == 0 or pd.isna(std):  # 处理常数列
    normalized = pd.Series(0, index=series.index)
    warning = f"警告: 列 '{column}' 是常数列或包含NaN值，将其替换为0"
  else:
    normalized = (series - mean) / std
    warning = None

  return column, normalized, warning


def minmax_normalize_column(df, column):
  """并行处理最小-最大归一化单列"""
  series = df[column]
  min_val = series.min()
  max_val = series.max()

  if max_val == min_val:  # 处理常数列
    normalized = pd.Series(0.5, index=series.index)
    warning = f"警告: 列 '{column}' 是常数列，将其设置为0.5"
  else:
    normalized = (series - min_val) / (max_val - min_val)
    warning = None

  return column, normalized, warning


def normalize_dataframe(df, method='zscore', n_jobs=-1):
  """并行归一化整个数据框
  
  Args:
      df: 要归一化的pandas DataFrame
      method: 'zscore'或'minmax'
      n_jobs: 并行进程数，-1表示使用所有可用核心
  
  Returns:
      normalized_df: 归一化后的DataFrame
  """
  columns = df.columns

  # 限制并行进程数不超过列数和可用核心数
  n_cores = multiprocessing.cpu_count()
  actual_jobs = min(n_jobs if n_jobs > 0 else n_cores, len(columns), n_cores)

  print(f"应用{method}归一化，并行处理 ({actual_jobs} 进程)...")

  if method == 'zscore':
    normalize_func = zscore_normalize_column
  else:  # minmax
    normalize_func = minmax_normalize_column

  # 并行处理所有列
  results = Parallel(n_jobs=actual_jobs)(
      delayed(normalize_func)(df, col) for col in columns
  )

  # 使用pd.concat一次性构建DataFrame，而非逐列添加，避免碎片化
  col_series_dict = {}
  warnings_list = []

  for col, normalized_series, warning in results:
    col_series_dict[col] = normalized_series
    if warning:
      warnings_list.append(warning)

  # 一次性创建DataFrame
  normalized_df = pd.DataFrame(col_series_dict, index=df.index)

  # 打印警告
  for warning in warnings_list:
    print(f"  {warning}")

  return normalized_df


def runTCDF(datafile, use_cache=False, normalization='zscore', n_jobs=-1):
  """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names.
  
  Args:
      datafile: 数据文件路径
      use_cache: 是否使用缓存
      normalization: 归一化方法，可选值为'zscore'(标准化)、'minmax'(最小-最大归一化)或None(不进行归一化)
      n_jobs: 并行处理的进程数，-1表示使用所有可用CPU核心
  """
  df_data = pd.read_csv(datafile)

  # 数据归一化，使用并行处理
  if normalization:
    df_data = normalize_dataframe(df_data, method=normalization, n_jobs=n_jobs)
    # 复制DataFrame以消除碎片
    df_data = df_data.copy()

  columns = list(df_data)

  allcauses = dict()
  alldelays = dict()
  allreallosses = dict()
  allscores = dict()

  # 为数据集生成缓存键
  dataset_cache_key = get_dataset_cache_key(datafile, args)
  if use_cache:
    # 在缓存键中也考虑归一化方法
    if normalization:
      dataset_cache_key += f"_norm_{normalization}"

    # 检查是否有已缓存的列
    dataset_cache = load_dataset_cache(dataset_cache_key)
    if dataset_cache and 'columns' in dataset_cache:
      print(f"找到数据集缓存: {datafile}")

  # 逐列处理
  for c in columns:
    idx = df_data.columns.get_loc(c)
    print(f"处理列 {idx+1}/{len(columns)}: {c}")

    # 检查该列是否有缓存
    column_data = None
    if use_cache and dataset_cache_key:
      column_data = load_column_cache(dataset_cache_key, idx, c)

    if column_data:
      print(f"  使用缓存的结果")
      causes = column_data['causes']
      # 将字符串键转回元组
      causeswithdelay = {}
      for k, v in column_data['causeswithdelay'].items():
        # 解析字符串形式的元组 "(a, b)" -> (a, b)
        key = eval(k)
        causeswithdelay[key] = v
      realloss = column_data['realloss']
      scores = column_data['scores']
    else:
      print(f"  计算因果关系...")
      # 创建临时CSV文件用于TCDF处理，包含归一化数据
      if normalization:
        temp_file = f"temp_{dataset_cache_key}_{idx}.csv"
        df_data.to_csv(temp_file, index=False)
        temp_datafile = temp_file
      else:
        temp_datafile = datafile

      causes, causeswithdelay, realloss, scores = TCDF.findcauses(
          c,
          cuda=cuda,
          epochs=nrepochs,
          kernel_size=kernel_size,
          layers=levels,
          log_interval=loginterval,
          lr=learningrate,
          optimizername=optimizername,
          seed=seed,
          dilation_c=dilation_c,
          significance=significance,
          file=temp_datafile
      )

      # 删除临时文件
      if normalization and os.path.exists(temp_file):
        os.remove(temp_file)

      # 立即保存该列的缓存
      if dataset_cache_key:
        save_column_cache(
            dataset_cache_key, idx, c, causes, causeswithdelay, realloss, scores
        )
        print(f"  已保存列缓存")

    # 汇总结果
    allscores[idx] = scores
    allcauses[idx] = causes
    alldelays.update(causeswithdelay)
    allreallosses[idx] = realloss

  return allcauses, alldelays, allreallosses, allscores, columns


def plotgraph(stringdatafile, alldelays, columns):
  """Plots a temporal causal graph showing all discovered causal relationships annotated with the time delay between cause and effect."""
  G = nx.DiGraph()
  for c in columns:
    G.add_node(c)
  for pair in alldelays:
    p1, p2 = pair
    nodepair = (columns[p2], columns[p1])

    G.add_edges_from([nodepair], weight=alldelays[pair])

  edge_labels = dict(
      [((
          u,
          v,
      ), d['weight']) for u, v, d in G.edges(data=True)]
  )

  pos = nx.circular_layout(G)
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
  nx.draw(
      G,
      pos,
      node_color='white',
      edge_color='black',
      node_size=1000,
      with_labels=True
  )
  ax = plt.gca()
  ax.collections[0].set_edgecolor("#000000")

  pylab.show()


def main(datafiles, evaluation):
  if evaluation:
    totalF1direct = []  #contains F1-scores of all datasets
    totalF1 = []  #contains F1'-scores of all datasets

    receptivefield = 1
    for l in range(0, levels):
      receptivefield += (kernel_size - 1) * dilation_c**(l)

  for datafile in datafiles.keys():
    stringdatafile = str(datafile)
    if '/' in stringdatafile:
      stringdatafile = str(datafile).rsplit('/', 1)[1]

    print("\n Dataset: ", stringdatafile)

    # run TCDF with cache option and normalization
    allcauses, alldelays, allreallosses, allscores, columns = runTCDF(
        datafile,
        use_cache=args.use_cache,
        normalization=args.normalization,
        n_jobs=args.n_jobs
    )

    print(
        "\n===================Results for", stringdatafile,
        "=================================="
    )
    for pair in alldelays:
      print(
          columns[pair[1]], "causes", columns[pair[0]], "with a delay of",
          alldelays[pair], "time steps."
      )

    if evaluation:
      # evaluate TCDF by comparing discovered causes with ground truth
      print(
          "\n===================Evaluation for", stringdatafile,
          "==============================="
      )
      FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct = evaluate(
          datafiles[datafile], allcauses, columns
      )
      totalF1.append(F1)
      totalF1direct.append(F1direct)

      # evaluate delay discovery
      extendeddelays, readgt, extendedreadgt = getextendeddelays(
          datafiles[datafile], columns
      )
      percentagecorrect = evaluatedelay(
          extendeddelays, alldelays, TPs, receptivefield
      ) * 100
      print(
          "Percentage of delays that are correctly discovered: ",
          percentagecorrect, "%"
      )

    print(
        "=================================================================================="
    )

    if args.plot:
      plotgraph(stringdatafile, alldelays, columns)

  # In case of multiple datasets, calculate average F1-score over all datasets and standard deviation
  if len(datafiles.keys()) > 1 and evaluation:
    print("\nOverall Evaluation: \n")
    print("F1' scores: ")
    for f in totalF1:
      print(f)
    print("Average F1': ", np.mean(totalF1))
    print("Standard Deviation F1': ", np.std(totalF1), "\n")
    print("F1 scores: ")
    for f in totalF1direct:
      print(f)
    print("Average F1: ", np.mean(totalF1direct))
    print("Standard Deviation F1: ", np.std(totalF1direct))


parser = argparse.ArgumentParser(
    description='TCDF: Temporal Causal Discovery Framework'
)

parser.add_argument(
    '--cuda',
    action="store_true",
    default=False,
    help='Use CUDA (GPU) (default: False)'
)
parser.add_argument(
    '--epochs',
    type=check_positive,
    default=1000,
    help='Number of epochs (default: 1000)'
)
parser.add_argument(
    '--kernel_size',
    type=check_positive,
    default=4,
    help=
    'Size of kernel, i.e. window size. Maximum delay to be found is kernel size - 1. Recommended to be equal to dilation coeffient (default: 4)'
)
parser.add_argument(
    '--hidden_layers',
    type=check_zero_or_positive,
    default=0,
    help='Number of hidden layers in the depthwise convolution (default: 0)'
)
parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.01,
    help='Learning rate (default: 0.01)'
)
parser.add_argument(
    '--optimizer',
    type=str,
    default='Adam',
    choices=['Adam', 'RMSprop'],
    help='Optimizer to use (default: Adam)'
)
parser.add_argument(
    '--log_interval',
    type=check_positive,
    default=500,
    help='Epoch interval to report loss (default: 500)'
)
parser.add_argument(
    '--seed',
    type=check_positive,
    default=1111,
    help='Random seed (default: 1111)'
)
parser.add_argument(
    '--dilation_coefficient',
    type=check_positive,
    default=4,
    help=
    'Dilation coefficient, recommended to be equal to kernel size (default: 4)'
)
parser.add_argument(
    '--significance',
    type=float,
    default=0.8,
    help=
    "Significance number stating when an increase in loss is significant enough to label a potential cause as true (validated) cause. See paper for more details (default: 0.8)"
)
parser.add_argument(
    '--plot',
    action="store_true",
    default=False,
    help='Show causal graph (default: False)'
)
parser.add_argument(
    '--use_cache',
    action="store_true",
    default=False,
    help='Use cached results if available (default: False)'
)
parser.add_argument(
    '--normalization',
    type=str,
    choices=['zscore', 'minmax', 'none'],
    default='zscore',
    help='Data normalization method: "zscore" (default), "minmax", or "none"'
)
parser.add_argument(
    '--n_jobs',
    type=int,
    default=-1,
    help=
    'Number of parallel jobs for normalization. -1 means using all available cores (default: -1)'
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument(
    '--ground_truth',
    action=StoreDictKeyPair,
    help=
    'Provide dataset(s) and the ground truth(s) to evaluate the results of TCDF. Argument format: DataFile1=GroundtruthFile1,Key2=Value2,... with a key for each dataset containing multivariate time series (required file format: csv, a column with header for each time series) and a value for the corresponding ground truth (required file format: csv, no header, index of cause in first column, index of effect in second column, time delay between cause and effect in third column)'
)
group.add_argument(
    '--data',
    nargs='+',
    help=
    '(Path to) one or more datasets to analyse by TCDF containing multiple time series. Required file format: csv with a column (incl. header) for each time series'
)

args = parser.parse_args()

print("Arguments:", args)

if torch.cuda.is_available():
  if not args.cuda:
    print(
        "WARNING: You have a CUDA device, you should probably run with --cuda to speed up training."
    )
if args.kernel_size != args.dilation_coefficient:
  print(
      "WARNING: The dilation coefficient is not equal to the kernel size. Multiple paths can lead to the same delays. Set kernel_size equal to dilation_c to have exaxtly one path for each delay."
  )

kernel_size = args.kernel_size
levels = args.hidden_layers + 1
nrepochs = args.epochs
learningrate = args.learning_rate
optimizername = args.optimizer
dilation_c = args.dilation_coefficient
loginterval = args.log_interval
seed = args.seed
cuda = args.cuda
significance = args.significance

if args.normalization.lower() == 'none':
  args.normalization = None

if args.ground_truth is not None:
  datafiles = args.ground_truth
  main(datafiles, evaluation=True)
else:
  datafiles = dict()
  for dataset in args.data:
    datafiles[dataset] = ""
  main(datafiles, evaluation=False)
