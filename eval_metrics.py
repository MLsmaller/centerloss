#-*- coding:utf-8 -*-
import operator
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from sklearn.metrics import roc_curve, auc
from IPython import embed

def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 30, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, distances,
        labels, nrof_folds=nrof_folds)
    thresholds = np.arange(0, 30, 0.001)
    val, val_std, far = calculate_val(thresholds, distances,
        labels, 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def calculate_roc(thresholds, distances, labels, nrof_folds=10):

    #---6000,一共有6000对验证的人脸,正负各3000
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    #---(10,3000)
    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)
    #---设置了0-30区间范围内3000个阈值，测试哪个阈值下的准确率较好
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #---len(train_set)->5400 len(test_set)  ->600
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        #---在训练集中选出一个最好的阈值然后在测试集上进行计算
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)
        #---这里的阈值设定的越来越大则tp越来越多，对应的tpr也会越来越高
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, distances[test_set], labels[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set], labels[test_set])
        #---对每一列求均值,即不同fold_idx下的结果求均值，tpr->(3000)-<len(threshold)
        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

def plot_acc(train_loss,val_acc,save_path='acc.png'):
    x=range(len(train_loss))
    plt.subplot(2,1,1)
    plt.plot(range(len(train_loss)),train_loss,color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')

    plt.subplot(2,1,2)
    plt.plot(range(len(val_acc)),val_acc,color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Val Accuracy')
    plt.savefig(save_path)

def plot_roc(fpr,tpr,figure_name="roc.png"):

    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='red',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    # embed()
    tp = np.sum(np.logical_and(predict_issame, actual_issame))  #---true positive  预测为正样本实际是正样本
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))  #---false positive 预测为正样本实际是负样本
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))  #---预测为负样本实际是负样本
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))  #---预测为负样本实际是正样本

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)   #---正确率
    #---召回率,和计算map差不多，随着阈值的增大，召回的越多，      #---假正率
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)  #---预测为正的样本数(实际为负)/实际的负样本数
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc



def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, distances[train_set], labels[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, distances[test_set], labels[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0,0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far