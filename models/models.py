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
        self.decoder = model2decoder[args.model](args)

    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return output

    def get_loss(self, outputs, data, split):
        raise NotImplementedError

    def compute_metrics(self, outputs, data, split):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError


class NCModel(BaseModel):
    # Base Model for Node Classification Task

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.n_classes = args.n_classes
        # Calculate Weight Matrix to balance samples
        data = args.data['y']
        pos = (data.long() == 1).float()
        neg = (data.long() == 0).float()
        alpha_pos = []
        alpha_neg = []
        for i in range(data.shape[1]):
            num_pos = torch.sum(data.long()[:, i] == 1).float()
            num_neg = torch.sum(data.long()[:, i] == 0).float()
            num_total = num_pos + num_neg
            alpha_pos.append(num_neg / num_total)
            alpha_neg.append(num_pos / num_total)
        alpha_pos = torch.Tensor([alpha_pos] * data.shape[0])
        alpha_neg = torch.Tensor([alpha_neg] * data.shape[0])
        self.weights = alpha_pos * pos + alpha_neg * neg
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def get_loss(self, outputs, data, split):
        idx = data[f'idx_{split}']
        outputs = outputs[idx]
        labels = data['y'][idx]
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float(), self.weights[idx])
        return loss

    def compute_metrics(self, outputs, data, split):
        idx = data[f'idx_{split}']
        outputs = outputs[idx]
        labels = data['y'][idx]
        acc, f1_micro, f1_macro, auc_micro, auc_macro, p5, r5 = nc_metrics(outputs, labels, self.n_classes)
        metrics = {'acc': acc, 'f1_micro': f1_micro, 'f1_macro': f1_macro,
                   'auc_micro': auc_micro, 'auc_macro': auc_macro, 'p@5': p5, 'r@5': r5}
        return metrics

    def has_improved(self, m1, m2):
        return (m1['f1_micro'] < m2['f1_micro']) \
               or (m1['f1_macro'] < m2['f1_macro']) \
               or (m1['auc_micro'] < m2['auc_micro']) \
               or (m1['auc_macro'] < m2['auc_macro'])

    def init_metric_dict(self):
        return {'acc': -1, 'f1_micro': -1, 'f1_macro': -1,
                   'auc_micro': -1, 'auc_macro': -1, 'p@5': -1, 'r@5': -1}
