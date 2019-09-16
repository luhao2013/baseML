"""
auc值的大小可以理解为: 随机抽一个正样本和一个负样本，正样本预测值比负样本大的概率
1.使用公式，排序后计算，O(NlogN)
2.使用近似方法，将预测值分桶，对正负样本分别构建直方图，再统计满足条件的正负样本对。
复杂度 O(N)
"""
def naive_auc(labels,preds):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total_pair = n_pos * n_neg

    labels_preds = zip(labels,preds)
    labels_preds = sorted(labels_preds,key=lambda x:x[1])
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(len(labels_preds)):
        if labels_preds[i][0] == 1:
            satisfied_pair += accumulated_neg
        else:
            accumulated_neg += 1

    return satisfied_pair / float(total_pair)


def roc_auc(labels,preds,n_bins=1000):
    # print(labels, preds)
    postive_len = sum(labels)
    negative_len = len(labels) - postive_len
    total_case = postive_len * negative_len
    pos_histogram = [0 for _ in range(n_bins)]
    neg_histogram = [0 for _ in range(n_bins)]
    bin_width = 1.0 / float(n_bins)
    for i in range(len(labels)):
        nth_bin = int(preds[i]/bin_width)
        if labels[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        # 该桶里每个正样本都大于accumulated_neg个负样本，对于桶内则大于一半的负样本
        satisfied_pair += (pos_histogram[i]*accumulated_neg + pos_histogram[i]*neg_histogram[i]*0.5)
        accumulated_neg += neg_histogram[i]
    return round(satisfied_pair / float(total_case), 2)


if __name__ == '__main__':
    # N = int(input())
    # labels, preds = [], []
    # for i in range(N):
    #     label, pred = map(float, input().split())
    #     labels.append(label)
    #     preds.append(pred)
    labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    preds = [0.90, 0.70, 0.60, 0.55, 0.52, 0.40, 0.38, 0.35, 0.31, 0.10]
    print(roc_auc(labels, preds))
    print(naive_auc(labels, preds))
