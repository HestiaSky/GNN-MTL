from layers.att_layers import *
from models.encoders import model2encoder
from models.decoders import model2decoder
from utils.eval_utils import *
from utils.ot_loss import *
from SinkhornOT import gw_iterative_1
import ot
import random


class BaseModel(nn.Module):
    # Base Model for KG Embedding Task

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.n_nodes = args.n_nodes
        self.device = args.device

    def get_neg(self, ILL, output, k):
        neg = []
        t = len(ILL)
        ILL_vec = np.array([output[e].detach().cpu().numpy() for e in ILL])
        KG_vec = np.array(output.detach().cpu())
        sim = scipy.spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
        for i in range(t):
            rank = sim[i, :].argsort()
            neg.append(rank[1:k+1])
        neg = np.array(neg)
        neg = neg.reshape((t * k,))
        return neg

    def get_neg_triplet(self, triples, head, tail, ids):
        neg = []
        for triple in triples:
            (h, r, t) = triple
            h2, r2, t2 = h, r, t
            neg_scope, num = True, 0
            while True:
                nt = random.randint(0, 999)
                if nt < 500:
                    if neg_scope:
                        h2 = random.sample(head[r], 1)[0]
                    else:
                        h2 = random.sample(range(ids), 1)[0]
                else:
                    if neg_scope:
                        t2 = random.sample(tail[r], 1)[0]
                    else:
                        t2 = random.sample(range(ids), 1)[0]
                if (h2, r2, t2) not in triples:
                    break
                else:
                    num += 1
                    if num > 10:
                        neg_scope = False
            neg.append((h2, r2, t2))
        return neg

    def compute_metrics(self, outputs, data, split):
        if split == 'train':
            pair = data['train']
        else:
            pair = data['test']
        if outputs.is_cuda:
            outputs = outputs.cpu()
        return get_hits(outputs, pair, top_k = [1])

    def has_improved(self, m1, m2):
        return (m1['Hits@10_l'] < m2['Hits@10_l']) \
               or (m1['Hits@10_r'] < m2['Hits@10_r'])

    def init_metric_dict(self):
        return {'Hits@1_l': -1, 'Hits@10_l': -1, 'Hits@50_l': -1, 'Hits@100_l': -1,
                'Hits@1_r': -1, 'Hits@10_r': -1, 'Hits@50_r': -1, 'Hits@100_r': -1}


class EAModel(BaseModel):
    # Base Model for Entity Alignment Task

    def __init__(self, args):
        super(EAModel, self).__init__(args)
        self.encoder = model2encoder[args.model](args)
        self.decoder = model2decoder[args.model](args)
        ILL = args.data['train']
        t = len(ILL)
        k = args.neg_num
        self.neg_num = k
        L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
        self.neg_left = L.reshape((t * k,))
        L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
        self.neg2_right = L.reshape((t * k,))
        self.neg_right = None
        self.neg2_left = None

    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return output

    def get_loss(self, outputs, data, split):
        ILL = data[split]
        left = ILL[:, 0]
        right = ILL[:, 1]
        t = len(ILL)
        k = self.neg_num
        left_x = outputs[left]
        right_x = outputs[right]
        A = torch.sum(torch.abs(left_x - right_x), 1)
        neg_l_x = outputs[self.neg_left]
        neg_r_x = outputs[self.neg_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        D = A + 1.0
        L1 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        neg_l_x = outputs[self.neg2_left]
        neg_r_x = outputs[self.neg2_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        L2 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        return (torch.sum(L1) + torch.sum(L2)) / (2.0 * t * k)


class UEAModel(BaseModel):
    # Base Model for Entity Alignment Task

    def __init__(self, args):
        super(UEAModel, self).__init__(args)
        self.ILL = None
        self.encoder = model2encoder[args.model](args)
        self.decoder = model2decoder[args.model](args)

    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return output

    def generate_pairs(self, outputs, data, bsz):
        e1, e2 = data['e1'], data['e2']
        index1, index2 = data['index1'], data['index2']
        outputs_arr = outputs.detach().cpu().numpy()
        L = np.array([index1[i] for i in range(e1)])
        R = np.array([index2[i] for i in range(e2)])
        M = scipy.spatial.distance.cdist(outputs_arr[L], outputs_arr[R], metric='cityblock')
        # left -> right
        v_l2r, idx_l2r = np.min(M, axis = 1), np.argmin(M, axis = 1)
        # right -> left
        v_r2l, idx_r2l = np.min(M, axis = 0), np.argmin(M, axis = 0)
        # intersection set
        pairs, scores = [], []
        for i, v in enumerate(idx_l2r):
            if idx_r2l[v] == i:
                pairs.append(np.array([i, v]))
                scores.append(v_l2r[i])
        pairs = np.array(pairs)
        # get the bsz first pairs ranking by scores
        print("generate {} pairs by the L1 distance".format(min(len(pairs), bsz)))
        if len(pairs) <= bsz:
            self.ILL = pairs
        idx = np.argsort(np.array(scores))[:bsz]
        self.ILL = pairs[idx]
        return
        
    def generate_neg(self, outputs, k):
        '''
            generate negative pairs according to IIL
            outputs: embedding space
            k: negative number
        '''
        t = len(self.ILL)
        self.neg_num = k
        L = np.ones((t, k)) * (self.ILL[:, 0].reshape((t, 1)))
        self.neg_left = L.reshape((t * k,))
        L = np.ones((t, k)) * (self.ILL[:, 1].reshape((t, 1)))
        self.neg2_right = L.reshape((t * k,))
        self.neg_right = self.get_neg(self.ILL[:, 0], outputs, k)
        self.neg2_left = self.get_neg(self.ILL[:, 1], outputs, k)
        return
    
    def get_loss(self, outputs):
        left = self.ILL[:, 0]
        right = self.ILL[:, 1]
        t = len(self.ILL)
        k = self.neg_num
        left_x = outputs[left]
        right_x = outputs[right]
        A = torch.sum(torch.abs(left_x - right_x), 1)
        neg_l_x = outputs[self.neg_left]
        neg_r_x = outputs[self.neg_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        D = A + 1.0
        L1 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        neg_l_x = outputs[self.neg2_left]
        neg_r_x = outputs[self.neg2_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        L2 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        return (torch.sum(L1) + torch.sum(L2)) / (2.0 * t * k)

    def get_loss_wassertein(self, outputs, data, bsz):
        # get the Wasserstein Distance between two graph node embedding spaces
        e1, e2 = data['e1'], data['e2']
        index1, index2 = data['index1'], data['index2']
        
        L = np.array([index1[i] for i in np.random.permutation(e1)[:bsz]])
        R = np.array([index2[i] for i in np.random.permutation(e2)[:bsz]])
        X, Y = outputs[L], outputs[R]

        # sinkhorn
        device = outputs.device
        a, b = torch.ones(bsz).to(device), torch.ones(bsz).to(device)
        M = torch.cdist(X, Y, p = 2)
        M = M.to(device)
        T, _ = sinkhorn(a, b, M.detach(), reg=0.01)
        newT = torch.zeros_like(T).to(device)
        newT[torch.arange(len(newT)), torch.argmax(newT, dim=1)] = 1

        return torch.sum(newT * M)

    def get_loss_gromove_wassertein(self, outputs, data, bsz):
        # get the Wasserstein Distance between two graph node embedding spaces
        e1, e2 = data['e1'], data['e2']
        index1, index2 = data['index1'], data['index2']
        L = np.array([index1[i] for i in np.random.permutation(e1)[:bsz]])
        R = np.array([index2[i] for i in np.random.permutation(e2)[:bsz]])
        X, Y = outputs[L], outputs[R]
        # entropic sinkhorn
        device = outputs.device
        a, b = torch.ones(bsz).to(device), torch.ones(bsz).to(device)
        M = torch.cdist(X, Y, p = 1)
        M = M.to(device)
        C1 = torch.cdist(X, X, p = 1).detach()
        C2 = torch.cdist(Y, Y, p = 1).detach()
        T, gwdist = gw_iterative_1(C1, C2, a, b, epsilon=0.01, max_iter=1000)
        newT = torch.zeros_like(T[0]).to(device)
        newT[torch.arange(len(newT)), torch.argmax(newT, dim=1)] = 1

        return torch.sum(newT * M)







