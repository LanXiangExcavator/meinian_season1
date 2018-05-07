# -*- coding: utf-8 -*-
"""
# @Time    :2018/5/7 下午10:39
# @Author  :Xuxian
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import time


def model_train(lgb_train, num_boost_round):
    params = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l2',
        'sub_feature': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,

        'min_data': 85,
        'max_depth': 14,
        'verbose': -1,
    }

    def fScore(preds, train_data):
        labels = train_data.get_label()
        a = np.log1p(preds) - np.log1p(labels)
        score = np.power(a, 2)
        return 'fScore', score.mean(), False

    gbm = lgb.train(params,
                    lgb_train,
                    feval=fScore,
                    valid_sets=[lgb_train],
                    num_boost_round=num_boost_round,
                    verbose_eval=10, )
    return gbm


def get_feature_and_target(feature, target, test):
    train_vid = target[['vid']]
    data = pd.merge(feature, train_vid, on='vid')
    data = pd.merge(data, target, 'left', on='vid')

    # 处理异常值
    data = data.drop(data[data['收缩压'].isnull()].index)

    data = data.drop(data[data['vid'] == '7685d48685028a006c84070f68854ce1'].index, axis=0)
    data = data.drop(data[data['vid'] == 'fa04c8db6d201b9f705a00c3086481b0'].index, axis=0)
    data = data.drop(data[data['vid'] == 'bd0322cf42fc6c2932be451e0b54ed02'].index, axis=0)
    data = data.drop(data[data['vid'] == 'de82a4130c4907cff4bfb96736674bbc'].index, axis=0)
    data = data.drop(data[data['vid'] == 'd9919661f0a45fbcacc4aa2c1119c3d2'].index, axis=0)
    data = data.drop(data[data['vid'] == '798d859a63044a8a5addf1f8c528629e'].index, axis=0)
    data_feature = data.drop(['vid', '收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白'], axis=1)
    data_target = data[['收缩压', '舒张压', '血清甘油三酯', '血清高密度脂蛋白', '血清低密度脂蛋白']]
    test_feature = pd.merge(feature, test[['vid']], on='vid')
    return data_feature, data_target, test_feature


def get_result(data_feature, data_target, test_feature, test):
    test_vid = test[['vid']]
    score = pd.DataFrame()
    score['vid'] = test_feature['vid']

    # 收缩压
    lgb_train = lgb.Dataset(data_feature, data_target['收缩压'])
    gbm = model_train(lgb_train, 350)
    score[0] = gbm.predict(test_feature.drop('vid', axis=1))

    # 舒张压
    lgb_train = lgb.Dataset(data_feature, data_target['舒张压'])
    gbm = model_train(lgb_train, 410)
    score[1] = gbm.predict(test_feature.drop('vid', axis=1))

    # 血清甘油三酯
    lgb_train = lgb.Dataset(data_feature, np.log(data_target['血清甘油三酯']))
    gbm = model_train(lgb_train, 230)
    score[2] = np.power(np.e, gbm.predict(test_feature.drop('vid', axis=1)))

    # 血清高密度脂蛋白
    lgb_train = lgb.Dataset(data_feature, data_target['血清高密度脂蛋白'])
    gbm = model_train(lgb_train, 590)
    score[3] = gbm.predict(test_feature.drop('vid', axis=1))

    # 血清低密度脂蛋白
    lgb_train = lgb.Dataset(data_feature, np.log(data_target['血清低密度脂蛋白']))
    gbm = model_train(lgb_train, 620)
    score[4] = np.power(np.e, gbm.predict(test_feature.drop('vid', axis=1)))

    result = pd.merge(test_vid, score, 'left', on='vid')
    result.to_csv('../submit/submit_' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.csv', header=False,
                  index=False)
