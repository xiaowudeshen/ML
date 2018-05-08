# -*- coding: utf-8 -*-
#!/bin/python
#xgboost的接口XGBClassifier - 是xgboost的sklearn包。这个包允许我们像GBM一样使用Grid Search 和并行处理
#from xgboost.sklearn import XGBClassifier  可以这样导入 XGBClassifier  参考https://blog.csdn.net/liulina603/article/details/78771738

import pickle
import xgboost as xgb
from sklearn.metrics import roc_curve

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris, load_digits, load_boston
from matplotlib import pyplot as plt

rng = np.random.RandomState(31337)

print("Zeros and Ones from the Digits dataset: binary classification")
digits = load_digits(2)
y = digits['target']
X = digits['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X,y):
    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict_proba(X[test_index])
    actuals = y[test_index]
#    print(predictions)
#    print(confusion_matrix(actuals, predictions))


print("Parameter optimization")
y = digits['target']
X = digits['data']
xgb_model = xgb.XGBClassifier(max_depth=4,
                        learning_rate=0.1,
                        n_estimators=5,
                        silent=True,
                        objective='binary:logistic',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=0.2,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=9999)

param_test1 = {'n_estimators': [90,100,110]}
clf = GridSearchCV(estimator=xgb_model,param_grid=param_test1, scoring='neg_log_loss', cv=3)
clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)
#print(clf.cv_results_)


# The sklearn API models are picklable
print("Pickling sklearn API models")
# must open in binary format to pickle
pickle.dump(clf, open("best_boston.pkl", "wb"))
clf2 = pickle.load(open("best_boston.pkl", "rb"))
# print(np.allclose(clf.predict(X), clf2.predict(X)))
# y_score = clf.predict_proba(X)[:,1]
# fpr,tpr,thresholds = roc_curve(y, y_score, pos_label=1)
# plt.plot([0,1],[0,1],'r--',fpr,tpr,'b')
# plt.show()
# Early-stopping

X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="error",
        eval_set=[(X_test, y_test)])

print(clf.feature_importances_)
xgb.plot_importance(clf)
xgb.plot_tree(clf)
plt.show()
