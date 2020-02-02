from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import torch
import scipy


def format_metrics(metrics, split):
    return " ".join(
        ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])


def acc_f1(output, labels, average='binary'):
    output = torch.sigmoid(output)
    if output.is_cuda:
        output = output.detach().cpu()
        labels = labels.detach().cpu()
    output = (output > 0.5).long()
    labels = labels.long()
    accuracy = accuracy_score(labels, output)
    precision = precision_score(labels, output, average=average)
    recall = recall_score(labels, output, average=average)
    f1 = f1_score(labels, output, average=average)
    return accuracy, precision, recall, f1


def nc_metrics(output, labels, n_classes):
    output = torch.sigmoid(output)
    if output.is_cuda:
        output = output.detach().cpu()
        labels = labels.detach().cpu()
    p5 = precision_at_k(output, labels, 5)
    r5 = recall_at_k(output, labels, 5)
    output = (output > 0.5).long()
    labels = labels.long()
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
    return f1_micro, f1_macro, auc_micro, auc_macro, p5, r5


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


def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1].detach().numpy() for e1, e2 in test_pair])
    Rvec = np.array([vec[e2].detach().numpy() for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1

    metrics = {}
    for i in range(len(top_lr)):
        metric_name, metric_val = 'Hits@{}_l'.format(top_k[i]), top_lr[i] / len(test_pair) * 100
        metrics[metric_name] = metric_val
    for i in range(len(top_rl)):
        metric_name, metric_val = 'Hits@{}_r'.format(top_k[i]), top_rl[i] / len(test_pair) * 100
        metrics[metric_name] = metric_val
    return metrics
