"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import math
import json

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def load_data(args):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats)
    elif args.task == 'lp':
        data = load_data_lp(args.dataset, args.use_feats)
    else:
        data = load_data_ea(args.dataset, args.use_feats)

    data['x'], data['adj'] = process(data['x'], data['adj'], args.normalize_x, args.normalize_adj)
    return data


# ############### FEATURES PROCESSING ############### #


def process(x, adj, norm_x, norm_adj):
    if norm_x:
        x = normalize(x)
    x = sparse_mx_to_torch_sparse_tensor(x)
    if norm_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return x, adj


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ############### Node Classification Dataloader ############### #


def load_data_nc(dataset, use_feats):
    if dataset in ['disc', 'disd', 'disp', 'med', 'dur']:
        names = ['y', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join("data/{}/{}_{}.pkl".format(dataset, dataset, names[i])), 'rb') as f:
                objects.append(pkl.load(f))
        y, graph = tuple(objects)
        all_idx = np.arange(len(y))
        np.random.shuffle(all_idx)
        all_idx = all_idx.tolist()
        nb_val = round(0.10 * len(all_idx))
        nb_test = round(0.20 * len(all_idx))
        idx_val, idx_test, idx_train = all_idx[:nb_val], all_idx[nb_val: nb_val + nb_test], all_idx[nb_val + nb_test:]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        if not use_feats:
            features = sp.coo_matrix(sp.eye(adj.shape[0]))
        y = torch.LongTensor(y)
        data = {'adj': adj, 'x': features, 'y': y, 'idx_train': idx_train, 'idx_val': idx_val,
                'idx_test': idx_test}

    elif dataset in ['multitask1', 'multitask2']:
        datapath = ['disc', 'med', 'dur']
        names = ['y', 'graph']
        objects = []
        for path in datapath:
            for i in range(len(names)):
                with open(os.path.join("data/{}/{}_{}.pkl".format(path, path, names[i])), 'rb') as f:
                    objects.append(pkl.load(f))
        y1, graph1, y2, graph2, y3, graph3 = tuple(objects)

        all_idx1 = np.arange(len(y1))
        np.random.shuffle(all_idx1)
        all_idx1 = all_idx1.tolist()
        nb_val = round(0.10 * len(all_idx1))
        nb_test = round(0.20 * len(all_idx1))
        idx_val1, idx_test1, idx_train1 = all_idx1[:nb_val], all_idx1[nb_val: nb_val + nb_test], all_idx1[
                                                                                                 nb_val + nb_test:]
        adj1 = nx.adjacency_matrix(nx.from_dict_of_lists(graph1))
        if not use_feats:
            features1 = sp.sparse.coo_matrix(sp.eye(adj1.shape[0]))
        y1 = torch.LongTensor(y1)

        all_idx2 = np.arange(len(y2))
        np.random.shuffle(all_idx2)
        all_idx2 = all_idx2.tolist()
        nb_val = round(0.10 * len(all_idx2))
        nb_test = round(0.20 * len(all_idx2))
        idx_val2, idx_test2, idx_train2 = all_idx2[:nb_val], all_idx2[nb_val: nb_val + nb_test], all_idx2[
                                                                                                 nb_val + nb_test:]
        adj2 = nx.adjacency_matrix(nx.from_dict_of_lists(graph2))
        if not use_feats:
            features2 = sp.sparse.coo_matrix(sp.eye(adj2.shape[0]))
        y2 = torch.LongTensor(y2)

        all_idx3 = np.arange(len(y3))
        np.random.shuffle(all_idx3)
        all_idx3 = all_idx3.tolist()
        nb_val = round(0.10 * len(all_idx3))
        nb_test = round(0.20 * len(all_idx3))
        idx_val3, idx_test3, idx_train3 = all_idx3[:nb_val], all_idx3[nb_val: nb_val + nb_test], all_idx3[
                                                                                                 nb_val + nb_test:]
        adj3 = nx.adjacency_matrix(nx.from_dict_of_lists(graph3))
        if not use_feats:
            features3 = sp.sparse.coo_matrix(sp.eye(adj3.shape[0]))
        y3 = torch.LongTensor(y3)

        data = {'adj1': adj1, 'x1': features1, 'y1': y1,
                'idx_train1': idx_train1, 'idx_val1': idx_val1, 'idx_test1': idx_test1,
                'adj2': adj2, 'x2': features2, 'y2': y2,
                'idx_train2': idx_train2, 'idx_val2': idx_val2, 'idx_test2': idx_test2,
                'adj3': adj3, 'x3': features3, 'y3': y3,
                'idx_train3': idx_train3, 'idx_val3': idx_val3, 'idx_tes3': idx_test3}
    else:
        data = None
    return data


# ############### Link Prediction Dataloader ############### #


# TODO: FB15K-237
def load_data_lp(dataset, use_feats):
    return None


# ############### Entity Alignment Dataloader ############### #


# calculate relation sets
def rfunc(e, KG):
    head = {}
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
            tail[tri[1]].add(tri[2])
    r_num = len(head)
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    for tri in KG:
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1

    return head, tail, head_r, tail_r


# get a dense adjacency matrix and degree
def get_matrix(e, KG):
    degree = [1] * e
    for tri in KG:
        if tri[0] != tri[2]:
            degree[tri[0]] += 1
            degree[tri[2]] += 1
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass
    for i in range(e):
        M[(i, i)] = 1
    return M, degree


# get a sparse tensor based on relational triples
def get_sparse_tensor(e, KG):
    print('getting a sparse tensor...')
    M, degree = get_matrix(e, KG)
    ind = []
    val = []
    for fir, sec in M:
        ind.append([fir, sec])
        val.append(M[(fir, sec)] / math.sqrt(degree[fir]) / math.sqrt(degree[sec]))
    M = torch.sparse.FloatTensor(torch.LongTensor(ind).t(), torch.FloatTensor(val), torch.Size([e, e]))
    return M


def get_features(lang):
    print('adding the primal input layer...')
    with open(file='data/dbp15k/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    ent_embeddings = torch.Tensor(embedding_list)
    return F.normalize(ent_embeddings, 2, 1)


# load a file and return a list of tuple containing $num integers in each line
def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def load_data_ea(dataset, use_feats):
    lang = dataset  # zh_en | ja_en | fr_en
    e1 = 'data/dbp15k/' + lang + '/ent_ids1'
    e2 = 'data/dbp15k/' + lang + '/ent_ids2'
    r1 = 'data/' + lang + '/rel_ids_1'
    r2 = 'data/' + lang + '/rel_ids_2'
    ill = 'data/' + lang + '/ref_ent_ids'
    ill_r = 'data/' + lang + '/ref_r_ids'
    kg1 = 'data/' + lang + '/triples_1'
    kg2 = 'data/' + lang + '/triples_2'

    e = len(set(loadfile(e1, 1)) | set(loadfile(e2, 1)))
    ILL = loadfile(ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * 3])
    test = ILL[illL // 10 * 3:]
    test_r = loadfile(ill_r, 2)
    KG = loadfile(kg1, 3) + loadfile(kg2, 3)

    features = get_features(lang[0:2])
    M = get_sparse_tensor(e, KG)
    head, tail, head_r, tail_r = rfunc(e, KG)
    data = {'x': features, 'adj': M, 'head': head, 'tail': tail, 'head_r': head_r, 'tail_r': tail_r,
            'train': train, 'test': test}
    return data


