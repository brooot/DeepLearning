import numpy as np
import matplotlib.pyplot as plt
from pylab import *

# 其中cm是计算好的混淆矩阵
# cm = confusion_matrix(test_label, predict_label)
# 比如上述这样产生cm
def ConfusionMatrixPng(cm,classlist):

    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.PuBu, interpolation='nearest')
    width = len(cm)
    height = len(cm[0])
    cb = fig.colorbar(res)
    alphabet = classlist
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    locs, labels = plt.xticks(range(width), alphabet[:width])
    for t in labels:
        t.set_rotation(45)
    # plt.xticks('orientation', 'vertical')
    # locs, labels = xticks([1,2,3,4], ['Frogs', 'Hogs', 'Bogs', 'Slogs'])
    # setp(alphabet, 'rotation', 'vertical')
    plt.xlabel("Predicted label",fontsize=10)
    plt.ylabel("True label")
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')
    print(norm_conf)

    for i in [0,1,2]:
        for j in [0,1,2]:
            # plt.text(i,j,"%.2f" % norm_conf[i][j],ha='center',fontsize=15)
            plt.text(i,j, "%.2f"%norm_conf[j][i], ha='center', fontsize=15)
    plt.title("Normalized confusion matrix")
    plt.show()


def confusion_matrix(test_label, predict_label):
    res = []
    for i in [1,2,3]:
        new_line = []
        for j in [1,2,3]:
            num = 0
            for k in range(len(test_label)):
                if i==test_label[k] and j==predict_label[k]:
                    num += 1
            new_line.append(num)
        res.append(new_line)
    # print("混淆矩阵是：")
    # print(res)
    return res


test_label    = [1,1,2,2,2,3,3,3,3]
predict_label = [1,2,2,2,3,3,3,1,2]
cm = confusion_matrix(test_label, predict_label)
classlist = ['1','2','3']

ConfusionMatrixPng(cm, classlist)