import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve, precision_recall_curve
from sklearn.externals import joblib

def fetch_test_data(filename):
    X = []
    y = []
    with open(filename) as f:
        for index,line in enumerate(f):
            line_list = line.strip().split(",")
            label = int(line_list[0])
            feature = list(map(float, line_list[1:]))
            y.append(label)
            X.append(feature)  # 这里要将数据转化成float类型
    X, y = np.array(X), np.array(y)
    return X, y

def plot_curve(y_test, y_pred):
    auc_score = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    precision, recall, thres = precision_recall_curve(y_test, y_pred)
    # plt.plot(recall,precision)
    F1 = [2 * p * r / (r + p) for (p, r) in zip(precision[:-1], recall[:-1])]
    F1_max_index = F1.index(max(F1))
    # print auc_score
    plt.figure(figsize=(10, 10), dpi=80)
    plt.figure(1)
    ax1 = plt.subplot(221)
    ax1.set_xlabel("fpr")
    ax1.set_ylabel("tpr  and auc_score=%f" % auc_score)
    plt.plot(fpr, tpr, color="r", linestyle="-")
    ax2 = plt.subplot(222)
    ax2.set_xlabel("thres")
    ax2.set_ylabel("precision and recall")
    plt.plot(thres, precision[:-1], '#8B0000', thres, recall[:-1], 'r--')
    ax3 = plt.subplot(223)
    ax3.set_xlabel("thres")
    ax3.set_ylabel("F1 and max_value's thres=%f" % thres[F1_max_index])
    plt.plot(thres, F1, '#9ACD32')
    # plt.plot([thres[F1_max_index],thres[F1_max_index]],[0,F1[F1_max_index]])
    ax4 = plt.subplot(224)
    ax4.set_xlabel("recall")
    ax4.set_ylabel("precision")
    plt.plot(recall, precision, '#9ACD32')
    plt.savefig("roc_F1_prec_recall.png")
    # plt.show()

def eval(y_test, y_pred):
    auc_score = roc_auc_score(y_test, y_pred)
    return auc_score

def process():
    test_file = "test_example"
    x_test, y_test = fetch_test_data(test_file)
    # model = linear_model.SGDClassifier(max_iter=1000, tol=1e-3,loss='log')
    model = joblib.load('lr_0.638899.model')
    y_pred = model.predict_proba(x_test)[:, 1]
    plot_curve(y_test, y_pred)
    


if __name__ == '__main__':
    process()



