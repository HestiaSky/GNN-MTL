import numpy as np
import torch 
import scipy.sparse as sp

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def torch_sparse_tensor_to_sparse_mx(sparse_tensor):
    """Convert a torch sparse tensor to a scipy sparse matrix."""
    data = sparse_tensor._values().numpy()
    row = sparse_tensor._indices().numpy()[0, :]
    col = sparse_tensor._indices().numpy()[1, :]
    return sp.coo_matrix((data,(row, col)), shape=(sparse_tensor.shape[0], sparse_tensor.shape[1]))

def loadfile(fn, num=1):
    """Load a file and return a list of tuple containing $num integers in each line."""
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

def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id

def loadattr(fns, e, ent2id):
    """The most frequent attributes are selected to save space."""
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    num_features = min(len(fre), 2000)
    attr2id = {}
    for i in range(num_features):
        attr2id[fre[i][0]] = i
    M = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            M[(ent2id[th[0]], attr2id[th[i]])] = 1.0
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[0])
        col.append(key[1])
        data.append(M[key])
    arr = sp.coo_matrix((data, (row, col)), shape=(e, num_features)) # attr
    return sparse_mx_to_torch_sparse_tensor(arr)

def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f

def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if

def get_weighted_adj(e, KG):
    r2f = func(KG)
    r2if = ifunc(KG)
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    adj =  sp.coo_matrix((data, (row, col)), shape=(e, e))
    return sparse_mx_to_torch_sparse_tensor(adj)

def load_data(args):
    names = [['ent_ids_1', 'ent_ids_2'], ['training_attrs_1', 'training_attrs_2'], ['triples_1', 'triples_2'], ['ref_ent_ids']]
    for fns in names:
        for i in range(len(fns)):
            fns[i] = 'data/'+args.lang+'/'+fns[i]
    Es, As, Ts, ill = names
    ill = ill[0]
    e = len(set(loadfile(Es[0], 1)) | set(loadfile(Es[1], 1)))
    ILL = loadfile(ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * args.split])
    test = ILL[illL // 10 * args.split:]
    KG = loadfile(Ts[0], 3) + loadfile(Ts[1], 3)
    ent2id = get_ent2id([Es[0], Es[1]])
    ae_input= loadattr([As[0], As[1]], e, ent2id)
    adj = get_weighted_adj(e, KG) # nx.adjacency_matrix(nx.from_dict_of_lists(get_dic_list(e, KG)))
    return adj, ae_input, train, test

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = torch_sparse_tensor_to_sparse_mx(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    # init_range = float(np.sqrt(6.0/(shape[0]+shape[1])))
    # with torch.no_grad():
    #     return torch.FloatTensor(shape[0], shape[1]).uniform_(-init_range, init_range)
    arr = np.load("layer2_init.npy")
    with torch.no_grad():
        tensor = torch.from_numpy(arr)
    return tensor 


def truncated_normal(shape):
    arr = np.load("layer1_init.npy")
    with torch.no_grad():
        tensor = torch.from_numpy(arr)
    return tensor 