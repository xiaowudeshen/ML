#!/bin/python

# -*- coding: utf-8 -*-
#xgboost原始接口进行交叉验证、训练，参数调整
import numpy as np
import xgboost as xgb
from sklearn.utils import shuffle

def read_file(filename):
    X_Y_data = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split("\t")
            X_Y_data.append(line)
    return np.array(X_Y_data)

### load data in do training
filename = "train_test_data"
x_y_data = read_file(filename)
x_y_data_1_list = []
x_y_data_0_list = []
for i in range(x_y_data.shape[0]):
    if x_y_data[i,-1]=='1':
        x_y_data_1_list.append(list(x_y_data[i,:]))
    else:
        x_y_data_0_list.append(list(x_y_data[i,:]))


x_y_data_1 = np.array(x_y_data_1_list)
x_y_data_0 = np.array(x_y_data_0_list)

x_y_data_1 = shuffle(x_y_data_1, random_state=1)
x_y_data_0 = shuffle(x_y_data_0, random_state=1)

x_y_data_1_train_sample = x_y_data_1[:250000,:]
x_y_data_1_test_sample = x_y_data_1[250000:,:]

x_y_data_0_train_sample = x_y_data_0[:800000,:]
x_y_data_0_test_sample = x_y_data_0[800000:,:]



train = shuffle(np.vstack((x_y_data_1_train_sample,x_y_data_0_train_sample)),random_state=1)
test = shuffle(np.vstack((x_y_data_1_test_sample,x_y_data_0_test_sample)),random_state=1)


dtrain = xgb.DMatrix(train[:,:-1],train[:,-1].astype(np.int32))
dtest = xgb.DMatrix(test[:,:-1],test[:,-1].astype(np.int32))
#param = {'n_estimators':100,'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
"""
num_round = 2

print('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'error'}, seed=0)

print('running cross validation, disable standard deviation display')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value
res = xgb.cv(param, dtrain, num_boost_round=10, nfold=5,
             metrics={'error'}, seed=0)
print(res)
"""
param = {'n_estimators':60,'max_depth':10, 'eta':1, 'silent':1, 'objective':'binary:logistic','booster':'gbtree','eval_metric':'logloss','scale_pos_weight':4}

evallist  = [(dtest,'eval'), (dtrain,'train')]
evals_result_dict = {}
num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist,evals_result=evals_result_dict)
print(evals_result_dict)

{'train': {'logloss': ['0.40904', '0.36686', '0.34983', '0.34172', '0.33773', '0.33382', '0.32961', '0.32300', '0.32042', '0.31817']},
  'eval': {'logloss': ['0.419153', '0.379281', '0.363893', '0.357078', '0.353992', '0.350062', '0.347147', '0.340415', '0.339358', '0.337345']}}
