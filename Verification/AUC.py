# -*- coding: utf-8 -*-  

"""
Created on 2021/4/6

@author: Ruoyu Chen
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from sklearn import metrics
from sklearn.metrics import auc
from prettytable import PrettyTable
        
def main(args):
    if os.path.exists(args.save_List):
        with open(args.save_List, "r") as f:
            datas = f.read().split('\n')
    else:
        raise ValueError("File {} not in path".format(args.save_List))
        
    score = []
    label = []
    for data in datas:
        try:
            score.append(float(data.split(' ')[2]))
            label.append(int(data.split(' ')[3]))
        except:
            pass
    score = np.array(score)
    label = np.array(label)
    
    x_labels = [10**-6, 10**-5, 10**-4,10**-3, 10**-2, 10**-1,0.2,0.4,0.6,0.8,1]
    tpr_fpr_table = PrettyTable(map(str, x_labels))

    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr) # select largest tpr at same fpr

    tpr_fpr_row = []
    
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(list(zip(abs(fpr-x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.4f' % tpr[min_index])
    tpr_fpr_table.add_row(tpr_fpr_row)

    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=1)

    plt.plot([0,1], [0,1], 'r', lw=1)

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    print(tpr_fpr_table)
    print("ROC AUC: {}".format(roc_auc))
    plt.savefig('AUC.jpg',dpi=400,bbox_inches='tight')
        # x = []
        # y = []
        # for i in np.linspace(0, 1, 1000):
        #     score_ = score.copy()
        #     score_[score_>=i] = 1
        #     score_[score_<i] = 0
        #     TP = sum(np.equal(score_[1001:2000],label[1001:2000]))
        #     FN = 1000 - TP
        #     TN = sum(np.equal(score_[0:1000],label[0:1000]))
        #     FP = 1000 - TN
        #     TPR = TP/(TP+FN)
        #     FPR = FP/(TN+FP)
        #     x.append(FPR)
        #     y.append(TPR)
        #     print(TPR)
        # plt.xlim([10**-6, 0.1])
        # plt.ylim([0.3, 1.0])
        # plt.xscale('log')
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.plot(x,y)
        
def parse_args():
    parser = argparse.ArgumentParser(description='Plot AUC curve')
    # general
    parser.add_argument('--save_List',
                        type=str,
                        default='./results/text/tutorial-case4.txt',
                        help='Datasets.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)