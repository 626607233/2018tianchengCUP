# 2018tianchengCUP
2018年甜橙金融杯大数据建模大学——DC竞赛_初赛

任务

通过训练学习用户在消费过程中的关联操作、交易详单信息，来识别交易风险。      



数据

*注 : 报名参赛或加入队伍后，可获取数据下载权限。
数据集共分为训练数据集、初赛测试数据集、复赛测试数据集。训练数据集中的文件包含黑白样本标签、用户交易详单、用户操作详单。初赛和复赛的测试集数据中则只包含用户交易详单、用户操作详单。
以下先简介数据集的文件名，然后再详述每个数据的所有字段。
 训练集数据：
1.	operation_train_new.csv为训练集操作详情表单，共1460843条数据；
2.	transaction_train_new.csv为训练集交易详情表单，共264654条数据；
3.	tag_train_new.csv为训练集黑白样本标签，共31179 条数据。

测试集初赛数据集：
1.	operation_round1_new.csv为初赛测试集操作详情表单，共1769049条数据；
2.	transaction_round1_new.csv为初赛测试集交易详情表单，共168981条数据；
注意：初赛测试集的日期数据（day）也是从1开始，但这里的日期1和训练集中的日期1不是同一天，而是指初赛测试集中的第一天的数据。（同理，请区分复赛测试集中的日期与其他数据集中的日期）

详细字段如下：
https://github.com/626607233/2018tianchengCUP/blob/master/交易详单数据字典.png

评分标准


评分算法

other

在黑产监控中，需要尽可能做到尽可能少的误伤和尽可能准确地探测，于是我们选择“在FPR较低时的TPR加权平均值”作为平均指标。

（FPR和TPR的定义请点击链接 ） 

 页末有前辈的代码
给定一个阀值，可根据混淆矩阵计算TPR（覆盖率）和FPR（打扰率）
TPR = TP /（TP + FN）
FPR = FP /（FP + TN）
其中，TP、FN、FP、TN分别为真正例、假反例、假正例、真反例。
通过设定不同的阈值，会有一系列TPR和FPR，就可以绘制出ROC曲线

这里的评分指标，首先计算了3个覆盖率TPR：
TPR1：FPR=0.001时的TPR
TPR2：FPR=0.005时的TPR
TPR3：FPR=0.01时的TPR
最终成绩= 0.4 * TPR1 + 0.3 * TPR2 + 0.3 * TPR3


def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3

