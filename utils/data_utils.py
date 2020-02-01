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
        data = load_data_ea(args.dataset)

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
    if dataset in ['dis', 'med', 'dur']:
        names = ['y', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join("data/mimic/{}/{}_{}.pkl".format(dataset, dataset, names[i])), 'rb') as f:
                objects.append(pkl.load(f))
        y, graph = tuple(objects)
        all_idx = np.arange(len(y))
        np.random.shuffle(all_idx)
        all_idx = all_idx.tolist()
        nb_val = round(0.10 * len(all_idx))
        nb_test = round(0.20 * len(all_idx))
        idx_val, idx_test, idx_train = all_idx[:nb_val], all_idx[nb_val: nb_val+nb_test], all_idx[nb_val+nb_test:]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        if not use_feats:
            features = sp.coo_matrix(sp.eye(adj.shape[0]))
        y = torch.LongTensor(y)
        data = {'adj': adj, 'x': features, 'y': y, 'idx_train': idx_train, 'idx_val': idx_val,
                'idx_test': idx_test}

    elif dataset in ['multitask1', 'multitask2']:
        datapath = ['dis', 'med', 'dur']
        names = ['id', 'y']
        objects = []
        for path in datapath:
            for i in range(len(names)):
                with open(os.path.join("data/mimic/multitask/{}_{}.pkl".format(path, names[i])), 'rb') as f:
                    objects.append(pkl.load(f))
        with open("data/mimic/multitask/graph.pkl", 'rb') as f:
            objects.append(pkl.load(f))
        dis_id, dis_y, med_id, med_y, dur_id, dur_y, graph = tuple(objects)

        all_idx = np.arange(len(dis_id))
        np.random.shuffle(all_idx)
        all_idx = all_idx.tolist()
        nb_val = round(0.10 * len(all_idx))
        nb_test = round(0.20 * len(all_idx))
        dis_val, dis_test, dis_train = all_idx[:nb_val], all_idx[nb_val: nb_val+nb_test], all_idx[nb_val+nb_test:]
        dis_y = torch.LongTensor(dis_y)

        all_idx = np.arange(len(med_id))
        np.random.shuffle(all_idx)
        all_idx = all_idx.tolist()
        nb_val = round(0.10 * len(all_idx))
        nb_test = round(0.20 * len(all_idx))
        med_val, med_test, med_train = all_idx[:nb_val], all_idx[nb_val: nb_val + nb_test], all_idx[nb_val + nb_test:]
        med_y = torch.LongTensor(med_y)

        all_idx = np.arange(len(dur_id))
        np.random.shuffle(all_idx)
        all_idx = all_idx.tolist()
        nb_val = round(0.10 * len(all_idx))
        nb_test = round(0.20 * len(all_idx))
        dur_val, dur_test, dur_train = all_idx[:nb_val], all_idx[nb_val: nb_val + nb_test], all_idx[nb_val + nb_test:]
        dur_y = torch.LongTensor(dur_y)

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        if not use_feats:
            features = sp.coo_matrix(sp.eye(graph.shape[0]))

        data = {'adj': adj, 'x': features, 'dis_id': dis_id, 'med_id': med_id, 'dur_id': dur_id,
                'dis_y': dis_y, 'dis_train': dis_train, 'dis_val': dis_val, 'dis_test': dis_test,
                'med_y': med_y, 'med_train': med_train, 'med_val': med_val, 'med_test': med_test,
                'dur_y': dur_y, 'dur_train': dur_train, 'dur_val': dur_val, 'dur_test': dur_test,
                }
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
    row = []
    col = []
    val = []
    for fir, sec in M:
        row.append(fir)
        col.append(sec)
        val.append(M[(fir, sec)] / math.sqrt(degree[fir]) / math.sqrt(degree[sec]))
    M = sp.coo_matrix((val, (row, col)), shape=(e, e))
    return M


def get_features(lang):
    print('adding the primal input layer...')
    with open(file='data/dbp15k/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    ent_embeddings = torch.Tensor(embedding_list)
    return sp.coo_matrix(F.normalize(ent_embeddings, 2, 1))


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


def load_data_ea(dataset):
    lang = dataset  # zh_en | ja_en | fr_en
    e1 = 'data/dbp15k/' + lang + '/ent_ids_1'
    e2 = 'data/dbp15k/' + lang + '/ent_ids_2'
    r1 = 'data/dbp15k/' + lang + '/rel_ids_1'
    r2 = 'data/dbp15k/' + lang + '/rel_ids_2'
    ill = 'data/dbp15k/' + lang + '/ref_ent_ids'
    ill_r = 'data/dbp15k/' + lang + '/ref_r_ids'
    kg1 = 'data/dbp15k/' + lang + '/triples_1'
    kg2 = 'data/dbp15k/' + lang + '/triples_2'

    e = len(set(loadfile(e1, 1)) | set(loadfile(e2, 1)))
    ILL = loadfile(ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * 3])
    test = np.array(ILL[illL // 10 * 3:])
    test_r = loadfile(ill_r, 2)
    KG = loadfile(kg1, 3) + loadfile(kg2, 3)

    features = get_features(lang[0:2])
    M = get_sparse_tensor(e, KG)
    head, tail, head_r, tail_r = rfunc(e, KG)
    data = {'x': features, 'adj': M, 'head': head, 'tail': tail, 'head_r': head_r, 'tail_r': tail_r,
            'train': train, 'test': test, 'test_r': test_r}
    return data


