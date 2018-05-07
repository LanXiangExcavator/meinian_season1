# -*- coding: utf-8 -*-
"""
# @Time    :2018/5/7 下午11:12
# @Author  :Xuxian
"""

from preProcess import *
from model import *
import pandas as pd

part_1 = pd.read_csv('../data/meinian_round1_data_part1_20180408.txt', sep='$')
part_2 = pd.read_csv('../data/meinian_round1_data_part2_20180408.txt', sep='$')
target = pd.read_csv('../data/meinian_round1_train_20180408.csv')
test = pd.read_csv('../data/meinian_round1_test_b_20180505.csv')
# 拼接数据
stitching(part_1, part_2)
part2_pivot(part_2)

data2 = pd.read_csv('../data/part2.csv')
data_all = pd.read_csv('../data/data_all.csv')

# 数值型特征
feature_1 = numeric_feature(part_2)

# 文本特征
feature_2 = process_text(data_all)

# 标签清洗
target = data_clean(target)
feature = feature = pd.merge(feature_1, feature_2, 'left', on='vid')

data_feature, data_target, test_feature = get_feature_and_target(feature, target, test)
get_result(data_feature, data_target, test_feature, test)
