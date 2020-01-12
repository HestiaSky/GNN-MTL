from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import numpy as np
import torch


def nc_metrics(output, labels, n_classes):
    output = torch.sigmoid(output)
    p5 = precision_at_k(output, labels, 5)
    r5 = recall_at_k(output, labels, 5)
    output = (output > 0.5).long()
    labels = labels.long()
    if output.is_cuda:
        output = output.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(labels, output)
    f1_micro = f1_score(labels, output, average='micro')
    f1_macro = f1_score(labels, output, average='macro')
    labels = np.array(labels)
    output = np.array(output)
    fpr, tpr, _ = roc_curve(labels.ravel(), output.ravel())
    auc_micro = auc(fpr, tpr)
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels[:, i], output[:, i])
        aucs.append(auc(fpr, tpr))
    auc_macro = np.mean(aucs)
    return accuracy, f1_micro, f1_macro, auc_micro, auc_macro, p5, r5


def precision_at_k(logits, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(logits)
    topk = sortd[:, -k:]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i, tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)


def recall_at_k(logits, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(logits)
    topk = sortd[:, -k:]

    #get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i, tk].sum()
        denom = y[i, :].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)