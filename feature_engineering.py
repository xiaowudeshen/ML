# -*- coding: utf-8 -*-
#参考文章https://mp.csdn.net/postedit/80019591
# 单变量特征选择
# 1.Pearson相关系数Pearson Correlation
import numpy as np
from scipy.stats import pearsonr

np.random.seed(0)
size = 300
x = np.random.normal(0, 1, size)
# Scipy的 pearsonr 方法能够同时计算相关系数和p-value，
# 比较了变量在加入噪音之前和之后的差异。当噪音比较小的时候，相关性很强，p-value很低
print("Lower noise", pearsonr(x, x + np.random.normal(0, 1, size)))
print("Higher noise", pearsonr(x, x + np.random.normal(0, 10, size)))

# 2.互信息和最大信息系数 Mutual information and maximal information coefficient (MIC)
"""
想把互信息直接用于特征选择其实不是太方便：1、它不属于度量方式，也没有办法归一化，在不同数据及上的结果无法做比较；2、对于连续变量的计算不是很方便（X和Y都是集合，x，y都是离散的取值），通常变量需要先离散化，而互信息的结果对离散化的方式很敏感。
最大信息系数克服了这两个问题。它首先寻找一种最优的离散化方式，然后把互信息取值转换成一种度量方式，取值区间在[0，1]。 minepy 提供了MIC功能。
反过头来看y=x^2这个例子，MIC算出来的互信息值为1(最大的取值)。
"""

# from minepy import MINE
# m = MINE()
# x = np.random.uniform(-1, 1, 10000)
# m.compute_score(x, x**2)
# print m.mic()
#
# from minepy import MINE
# m = MINE()
# x = np.random.uniform(-1, 1, 10000)
# m.compute_score(x, x**2)
# print(m.mic())

# 3基于学习模型的特征排序
"""
基于树的方法比较易于使用，因为他们对非线性关系的建模比较好，并且不需要太多的调试。但要注意过拟合问题，因此树的深度最好不要太大，再就是运用交叉验证。
在 波士顿房价数据集 上使用sklearn的 随机森林回归 给出一个单变量选择的例子：
"""
# from sklearn.model_selection import cross_val_score, ShuffleSplit
# from sklearn.datasets import load_boston
# from sklearn.ensemble import RandomForestRegressor
#
# #Load boston housing dataset as an example
# boston = load_boston()
# X = boston["data"]
# Y = boston["target"]
# names = boston["feature_names"]
#
# rf = RandomForestRegressor(n_estimators=20, max_depth=4)
# scores = []
# for i in range(X.shape[1]):
#      score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",cv=ShuffleSplit(len(X), train_size=.8, test_size=.2))
#      scores.append((round(np.mean(score), 3), names[i]))
# print(sorted(scores, reverse=True))

# 线性模型和正则化
"""
单变量特征选择方法独立的衡量每个特征与响应变量之间的关系，另一种主流的特征选择方法是基于机器学习模型的方法。有些机器学习方法本身就具有对特征进行打分的机制，或者很容易将其运用到特征选择任务中，例如回归模型，SVM，决策树，随机森林等等。说句题外话，这种方法好像在一些地方叫做wrapper类型，大概意思是说，特征排序模型和机器学习模型是耦盒在一起的，对应的非wrapper类型的特征选择方法叫做filter类型。
下面将介绍如何用回归模型的系数来选择特征。越是重要的特征在模型中对应的系数就会越大，而跟输出变量越是无关的特征对应的系数就会越接近于0。在噪音不多的数据上，或者是数据量远远大于特征数的数据上，如果特征之间相对来说是比较独立的，那么即便是运用最简单的线性回归模型也一样能取得非常好的效果。
"""


# from sklearn.linear_model import LinearRegression
# import numpy as np
# np.random.seed(0)
# size = 5000
# #A dataset with 3 features
# X = np.random.normal(0, 1, (size, 3))
# #Y = X0 + 2*X1 + noise
# Y = X[:,0] + 2*X[:,1] + np.random.normal(0, 2, size)
# lr = LinearRegression()
# lr.fit(X, Y)
# A helper method for pretty-printing linear models
def pretty_print_linear(coefs, names=None, sort=False):
    if names is None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)


# print("Linear model:", pretty_print_linear(lr.coef_))

# 在很多实际的数据当中，往往存在多个互相关联的特征，这时候模型就会变得不稳定，数据中细微的变化就可能导致模型的巨大变化（模型的变化本质上是系数，或者叫参数，可以理解成W），这会让模型的预测变得困难，这种现象也称为多重共线性。例如，假设我们有个数据集，它的真实模型应该是Y=X1+X2，当我们观察的时候，发现Y’=X1+X2+e，e是噪音。如果X1和X2之间存在线性关系，例如X1约等于X2，这个时候由于噪音e的存在，我们学到的模型可能就不是Y=X1+X2了，有可能是Y=2X1，或者Y=-X1+3X2。下边这个例子当中，在同一个数据上加入了一些噪音，用随机森林算法进行特征选择。
# from sklearn.linear_model import LinearRegression
#
# size = 100
# np.random.seed(seed=5)
#
# X_seed = np.random.normal(0, 1, size)
# X1 = X_seed + np.random.normal(0, .1, size)
# X2 = X_seed + np.random.normal(0, .1, size)
# X3 = X_seed + np.random.normal(0, .1, size)
#
# Y = X1 + X2 + X3 + np.random.normal(0,1, size)
# X = np.array([X1, X2, X3]).T
#
# lr = LinearRegression()
# lr.fit(X,Y)
# print("Linear model:", pretty_print_linear(lr.coef_))
"""
系数之和接近3，基本上和上上个例子的结果一致，应该说学到的模型对于预测来说还是不错的。但是，如果从系数的字面意思上去解释特征的重要性的话，X3对于输出变量来说具有很强的正面影响，而X1具有负面影响，而实际上所有特征与输出变量之间的影响是均等的。
同样的方法和套路可以用到类似的线性模型上，比如逻辑回归。
"""
# 3.1正则化模型
"""
正则化就是把额外的约束或者惩罚项加到已有模型（损失函数）上，以防止过拟合并提高泛化能力。损失函数由原来的E(X,Y)变为E(X,Y)+alpha||w||，w是模型系数组成的向量（有些地方也叫参数parameter，coefficients），||·||一般是L1或者L2范数，alpha是一个可调的参数，控制着正则化的强度。当用在线性模型上时，L1正则化和L2正则化也称为Lasso和Ridge。
"""
# 3.2L1正则化/Lasso
"""
L1正则化将系数w的l1范数作为惩罚项加到损失函数上，由于正则项非零，这就迫使那些弱的特征所对应的系数变成0。因此L1正则化往往会使学到的模型很稀疏（系数w经常为0），这个特性使得L1正则化成为一种很好的特征选择方法。
Scikit-learn为线性回归提供了Lasso，为分类提供了L1逻辑回归。
下面的例子在波士顿房价数据上运行了Lasso，其中参数alpha是通过grid search进行优化的。
"""
# from sklearn.linear_model import Lasso
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import load_boston
#
#
# boston = load_boston()
# scaler = StandardScaler()
# X = scaler.fit_transform(boston["data"])
# Y = boston["target"]
# names = boston["feature_names"]
#
# lasso = Lasso(alpha=.3)
# lasso.fit(X, Y)
#
# print("Lasso model: ", pretty_print_linear(lasso.coef_, names, sort = True))

"""
可以看到，很多特征的系数都是0。如果继续增加alpha的值，得到的模型就会越来越稀疏，即越来越多的特征系数会变成0。
然而，L1正则化像非正则化线性模型一样也是不稳定的，如果特征集合中具有相关联的特征，当数据发生细微变化时也有可能导致很大的模型差异。
"""
# 3.3 L2正则化/Ridge regression
"""
L2正则化将系数向量的L2范数添加到了损失函数中。由于L2惩罚项中系数是二次方的，这使得L2和L1有着诸多差异，最明显的一点就是，L2正则化会让系数的取值变得平均。对于关联特征，这意味着他们能够获得更相近的对应系数。还是以Y=X1+X2为例，假设X1和X2具有很强的关联，如果用L1正则化，不论学到的模型是Y=X1+X2还是Y=2X1，惩罚都是一样的，都是2alpha。但是对于L2来说，第一个模型的惩罚项是2 alpha，但第二个模型的是4*alpha。可以看出，系数之和为常数时，各系数相等时惩罚是最小的，所以才有了L2会让各个系数趋于相同的特点。
可以看出，L2正则化对于特征选择来说一种稳定的模型，不像L1正则化那样，系数会因为细微的数据变化而波动。所以L2正则化和L1正则化提供的价值是不同的，L2正则化对于特征理解来说更加有用：表示能力强的特征对应的系数是非零。
回过头来看看3个互相关联的特征的例子，分别以10个不同的种子随机初始化运行10次，来观察L1和L2正则化的稳定性
"""
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score

size = 100
# We run the method 10 times with different random seeds
for i in range(10):
    print("Random seed %s" % i)
    np.random.seed(seed=i)
    X_seed = np.random.normal(0, 1, size)
    X1 = X_seed + np.random.normal(0, .1, size)
    X2 = X_seed + np.random.normal(0, .1, size)
    X3 = X_seed + np.random.normal(0, .1, size)
    Y = X1 + X2 + X3 + np.random.normal(0, 1, size)
    X = np.array([X1, X2, X3]).T
    lr = LinearRegression()
    lr.fit(X, Y)
    print("Linear model:", pretty_print_linear(lr.coef_))
    lasso = Lasso(alpha=.10)
    lasso.fit(X, Y)
    print("Lasso model: ", pretty_print_linear(lasso.coef_))

    ridge = Ridge(alpha=10)
    ridge.fit(X, Y)
    print("Ridge model:", pretty_print_linear(ridge.coef_))
    """
    Random seed 0
Linear model: 0.728 * X0 + 2.309 * X1 + -0.082 * X2
Lasso model:  0.683 * X0 + 2.177 * X1 + 0.0 * X2
Ridge model: 0.938 * X0 + 1.059 * X1 + 0.877 * X2
Random seed 1
Linear model: 1.152 * X0 + 2.366 * X1 + -0.599 * X2
Lasso model:  0.854 * X0 + 1.951 * X1 + 0.0 * X2
Ridge model: 0.984 * X0 + 1.068 * X1 + 0.759 * X2
Random seed 2
Linear model: 0.697 * X0 + 0.322 * X1 + 2.086 * X2
Lasso model:  0.764 * X0 + 0.267 * X1 + 1.978 * X2
Ridge model: 0.972 * X0 + 0.943 * X1 + 1.085 * X2
Random seed 3
Linear model: 0.287 * X0 + 1.254 * X1 + 1.491 * X2
Lasso model:  0.081 * X0 + 1.321 * X1 + 1.537 * X2
Ridge model: 0.919 * X0 + 1.005 * X1 + 1.033 * X2
Random seed 4
Linear model: 0.187 * X0 + 0.772 * X1 + 2.189 * X2
Lasso model:  0.206 * X0 + 0.651 * X1 + 2.187 * X2
Ridge model: 0.964 * X0 + 0.982 * X1 + 1.098 * X2
Random seed 5
Linear model: -1.291 * X0 + 1.591 * X1 + 2.747 * X2
Lasso model:  0.0 * X0 + 0.746 * X1 + 2.175 * X2
Ridge model: 0.758 * X0 + 1.011 * X1 + 1.139 * X2
Random seed 6
Linear model: 1.199 * X0 + -0.031 * X1 + 1.915 * X2
Lasso model:  1.121 * X0 + 0.0 * X1 + 1.863 * X2
Ridge model: 1.016 * X0 + 0.89 * X1 + 1.091 * X2
Random seed 7
Linear model: 1.474 * X0 + 1.762 * X1 + -0.151 * X2
Lasso model:  1.38 * X0 + 1.591 * X1 + 0.012 * X2
Ridge model: 1.018 * X0 + 1.039 * X1 + 0.901 * X2
Random seed 8
Linear model: 0.084 * X0 + 1.88 * X1 + 1.107 * X2
Lasso model:  0.108 * X0 + 1.94 * X1 + 0.94 * X2
Ridge model: 0.907 * X0 + 1.071 * X1 + 1.008 * X2
Random seed 9
Linear model: 0.714 * X0 + 0.776 * X1 + 1.364 * X2
Lasso model:  0.612 * X0 + 0.562 * X1 + 1.579 * X2
Ridge model: 0.896 * X0 + 0.903 * X1 + 0.98 * X2
    可以看出，不同的数据上线性回归得到的模型（系数）相差甚远，但对于L2正则化模型来说，结果中的系数非常的稳定，差别较小，都比较接近于1，能够反映出数据的内在结构。
    """

# 4 递归特征消除 Recursive feature elimination (RFE)
"""
递归特征消除的主要思想是反复的构建模型（如SVM或者回归模型）然后选出最好的（或者最差的）的特征（可以根据系数来选），把选出来的特征放到一遍，然后在剩余的特征上重复这个过程，直到所有特征都遍历了。这个过程中特征被消除的次序就是特征的排序。因此，这是一种寻找最优特征子集的贪心算法。
RFE的稳定性很大程度上取决于在迭代的时候底层用哪种模型。例如，假如RFE采用的普通的回归，没有经过正则化的回归是不稳定的，那么RFE就是不稳定的；假如采用的是Ridge，而用Ridge正则化的回归是稳定的，那么RFE就是稳定的。
Sklearn提供了 RFE 包，可以用于特征消除，还提供了 RFECV ，可以通过交叉验证来对的特征进行排序。
"""
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

# use linear regression as the model
lr = LinearRegression()
# rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(X, Y)

print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

# 完整的例子
"""
X1到X5是由 单变量分布 生成的，e是 标准正态变量 N(0,1)。另外，原始的数据集中含有5个噪音变量 X5,…,X10，跟响应变量是独立的。我们增加了4个额外的变量X11,…X14，分别是X1,…,X4的关联变量，通过f(x)=x+N(0,0.01)生成，这将产生大于0.999的关联系数。这样生成的数据能够体现出不同的特征排序方法应对关联特征时的表现。
接下来将会在上述数据上运行所有的特征选择方法，并且将每种方法给出的得分进行归一化，让取值都落在0-1之间。对于RFE来说，由于它给出的是顺序而不是得分，我们将最好的5个的得分定为1，其他的特征的得分均匀的分布在0-1之间。
"""
from sklearn.datasets import load_boston
from sklearn.linear_model import (LinearRegression, Ridge,
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from minepy import MINE

np.random.seed(0)
size = 750
X = np.random.uniform(0, 1, (size, 14))
# "Friedamn #1” regression problem
Y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 +
     10 * X[:, 3] + 5 * X[:, 4] + np.random.normal(0, 1))
# Add 3 additional correlated variables (correlated with X1-X3)
X[:, 10:] = X[:, :4] + np.random.normal(0, .025, (size, 4))
names = ["x%s" % i for i in range(1, 15)]
ranks = {}


def rank_to_dict(ranks, names, order=1):
    ranks = np.array(ranks).reshape(1,-1)
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array(ranks).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))


lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)
ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)
# stop the search when 5 features are left (they will get equal scores)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X, Y)
rfe_feature = [ele for ele in map(float, rfe.ranking_)]
ranks["RFE"] = rank_to_dict(rfe_feature, names, order=-1)
rf = RandomForestRegressor()
rf.fit(X,Y)
ranks["RF"] = rank_to_dict(rf.feature_importances_, names)
f, pval  = f_regression(X, Y, center=True)
ranks["Corr."] = rank_to_dict(f, names)
mine = MINE()
mic_scores = []
for i in range(X.shape[1]):
	mine.compute_score(X[:,i], Y)
	m = mine.mic()
	mic_scores.append(m)
ranks["MIC"] = rank_to_dict(mic_scores, names)
r = {}
for name in names:
	r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
print("\t%s" % "\t".join(methods))
for name in names:
	print("%s\t%s" % (name, "\t".join(map(str,[ranks[method][name] for method in methods]))))

"""
	Corr.	Lasso	Linear  reg	    MIC	RF	RFE	    Ridge	Stability	Mean
x1	0.3	    0.79	1.0	    0.39	0.55	1.0	    0.77	0.77	    0.7
x2	0.44	0.83	0.56	0.61	0.67	1.0	    0.75	0.72	    0.7
x3	0.0	    0.0	    0.5 	0.34	0.13	1.0	    0.05	0.0	        0.25
x4	1.0	    1.0	    0.57	1.0	    0.56	1.0	    1.0	    1.0	        0.89
x5	0.1	    0.51	0.27	0.2	    0.29	0.78	0.88	0.55	    0.45
x6	0.0	    0.0	    0.02	0.0	    0.01	0.44	0.05	0.0	        0.06
x7	0.01	0.0	    0.0	    0.07	0.02	0.0	    0.01	0.0	        0.01
x8	0.02	0.0	    0.03	0.05	0.01	0.56	0.09	0.0	        0.1
x9	0.01	0.0	    0.0	    0.09	0.01	0.11	0.0	    0.0	        0.03
x10	0.0	    0.0	    0.01	0.04	0.0	    0.33	0.01	0.0	        0.05
x11	0.29	0.0	    0.6	    0.43	0.39	1.0	    0.59	0.37	    0.46
x12	0.44	0.0	    0.14	0.71	0.35	0.67	0.68	0.47	    0.43
x13	0.0	    0.0	    0.48	0.23	0.07	0.89	0.02	0.0	        0.21
x14	0.99	0.16	0.0	    1.0	    1.0 	0.22	0.95	0.62	    0.62
特征之间存在 线性关联 关系，每个特征都是独立评价的，因此X1,…X4的得分和X11,…X14的得分非常接近，而噪音特征X5,…,X10正如预期的那样和响应变量之间几乎没有关系。由于变量X3是二次的，因此X3和响应变量之间看不出有关系（除了MIC之外，其他方法都找不到关系）。这种方法能够衡量出特征和响应变量之间的线性关系，但若想选出优质特征来提升模型的泛化能力，这种方法就不是特别给力了，因为所有的优质特征都不可避免的会被挑出来两次。
Lasso能够挑出一些优质特征，同时让其他特征的系数趋于0。当如需要减少特征数的时候它很有用，但是对于数据理解来说不是很好用。（例如在结果表中，X11,X12,X13的得分都是0，好像他们跟输出变量之间没有很强的联系，但实际上不是这样的）
MIC对特征一视同仁，这一点上和关联系数有点像，另外，它能够找出X3和响应变量之间的非线性关系。
随机森林基于不纯度的排序结果非常鲜明，在得分最高的几个特征之后的特征，得分急剧的下降。从表中可以看到，得分第三的特征比第一的小4倍。而其他的特征选择算法就没有下降的这么剧烈。
Ridge将回归系数均匀的分摊到各个关联变量上，从表中可以看出，X11,…,X14和X1,…,X4的得分非常接近。
稳定性选择常常是一种既能够有助于理解数据又能够挑出优质特征的这种选择，在结果表中就能很好的看出。像Lasso一样，它能找到那些性能比较好的特征（X1，X2，X4，X5），同时，与这些特征关联度很强的变量也得到了较高的得分。
"""
