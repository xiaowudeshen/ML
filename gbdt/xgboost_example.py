from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from xgboost import plot_importance
import matplotlib.pyplot as plt


breast_cancer = datasets.load_breast_cancer()
data, target = breast_cancer['data'], breast_cancer['target']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


xgb_model = XGBClassifier(objective='binary:logistic',booster='gbtree')
parameters = {'n_estimators':[1,10,20,30,40,50,60],'learning_rate': [0.01, 0.02, 0.03], 'max_depth': [4, 5, 6], 'min_child_weight':[1,2,3,4,5,6]}
clf = GridSearchCV(xgb_model, parameters, scoring='roc_auc',verbose=1,cv=5)
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_params_)

xgb_model = XGBClassifier(n_estimators=clf.best_params_['n_estimators'],learning_rate=clf.best_params_['learning_rate'],max_depth=clf.best_params_['max_depth'], min_child_weight=clf.best_params_['min_child_weight'])
xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc",eval_set=[(X_test, y_test)])
preds = xgb_model.predict_proba(X_test)
print(preds[:,1])
auc = roc_auc_score(y_test, preds[:,1])
print(auc)


fig,ax = plt.subplots(figsize=(15,15))
plot_importance(xgb_model,height=0.5,ax=ax,max_num_features=64)
plt.show()
