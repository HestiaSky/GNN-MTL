from layers.att_layers import *
from models.encoders import model2encoder
from models.decoders import model2decoder
from utils.eval_utils import *


class BaseModel(nn.Module):
    # Base Model for KG Embedding Task

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.n_nodes = args.n_nodes
        self.encoder = model2encoder[args.model](args)

    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def get_loss(self, outputs, data, split):
        raise NotImplementedError

    def compute_metrics(self, outputs, data, split):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError


class EAModel(BaseModel):
    # Base Model for Entity Alignment Task

    def __init__(self, args):
        super(EAModel, self).__init__(args)
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

    def get_neg(self, ILL, output, k):
        neg = []
        t = len(ILL)
        ILL_vec = np.array([output[e].detach().cpu().numpy() for e in ILL])
        KG_vec = np.array(output.detach().cpu())
        sim = scipy.spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
        for i in range(t):
            rank = sim[i, :].argsort()
            neg.append(rank[0:k])
        neg = np.array(neg)
        neg = neg.reshape((t * k,))
        return neg

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
        D = A + 1.0
        neg_l_x = outputs[self.neg_left]
        neg_r_x = outputs[self.neg_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        L1 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        neg_l_x = outputs[self.neg2_left]
        neg_r_x = outputs[self.neg2_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        L2 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        return (torch.sum(L1) + torch.sum(L2)) / (2.0 * t * k)

    def compute_metrics(self, outputs, data, split):
        if split == 'train':
            pair = data['train']
        else:
            pair = data['test']
        if outputs.is_cuda:
            outputs = outputs.cpu()
        return get_hits(outputs, pair)

    def has_improved(self, m1, m2):
        return (m1['Hits@10_l'] < m2['Hits@10_l']) \
               or (m1['Hits@100_l'] < m2['Hits@100_l']) \
               or (m1['Hits@10_r'] < m2['Hits@10_r']) \
               or (m1['Hits@100_r'] < m2['Hits@100_r'])

    def init_metric_dict(self):
        return {'Hits@1_l': -1, 'Hits@10_l': -1, 'Hits@50_l': -1, 'Hits@100_l': -1,
                'Hits@1_r': -1, 'Hits@10_r': -1, 'Hits@50_r': -1, 'Hits@100_r': -1}
