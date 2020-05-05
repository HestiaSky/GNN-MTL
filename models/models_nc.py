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


class NCModel(BaseModel):
    # Base Model for Node Classification Task

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.n_classes = args.n_classes
        self.decoder = model2decoder[args.model](args)
        # Calculate Weight Matrix to balance samples
        data = args.data['y']
        self.weights = self.get_weights(data)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def get_weights(self, data):
        pos = (data.long() == 1).float()
        neg = (data.long() == 0).float()
        alpha_pos = []
        alpha_neg = []
        if len(data.shape) > 1:
            for i in range(data.shape[1]):
                num_pos = torch.sum(data.long()[:, i] == 1).float()
                num_neg = torch.sum(data.long()[:, i] == 0).float()
                num_total = num_pos + num_neg
                alpha_pos.append(num_neg / num_total)
                alpha_neg.append(num_pos / num_total)
        else:
            num_pos = torch.sum(data.long() == 1).float()
            num_neg = torch.sum(data.long() == 0).float()
            num_total = num_pos + num_neg
            alpha_pos = num_neg / num_total
            alpha_neg = num_pos / num_total
        alpha_pos = torch.Tensor([alpha_pos] * data.shape[0])
        alpha_neg = torch.Tensor([alpha_neg] * data.shape[0])
        return alpha_pos * pos + alpha_neg * neg

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return output

    def get_loss(self, outputs, data, split):
        idx = data[f'idx_{split}']
        outputs = outputs[idx]
        labels = data['y'][idx]
        if self.n_classes == 1:
            outputs = outputs.view(-1)
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float(), self.weights[idx])
        return loss

    def compute_metrics(self, outputs, data, split):
        outputs = torch.sigmoid(outputs)
        if outputs.is_cuda:
            outputs = outputs.detach().cpu()
            labels = data['y'].detach().cpu()
        idx = data[f'idx_{split}']
        outputs = outputs[idx]
        labels = labels[idx]
        if self.n_classes > 1:
            f1_micro, f1_macro, auc_micro, auc_macro, p5, r5 = nc_metrics(outputs, labels)
            metrics = {'f1_micro': f1_micro, 'f1_macro': f1_macro,
                       'auc_micro': auc_micro, 'auc_macro': auc_macro, 'p@5': p5, 'r@5': r5}
        else:
            acc, pre, rec, f1, auc = acc_f1(outputs, labels)
            metrics = {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1, 'auc': auc}
        return metrics

    def has_improved(self, m1, m2):
        if self.n_classes > 1:
            return m1['auc_macro'] < m2['auc_macro']
        else:
            return m1['auc'] < m2['auc']

    def init_metric_dict(self):
        if self.n_classes > 1:
            return {'f1_micro': -1, 'f1_macro': -1,
                    'auc_micro': -1, 'auc_macro': -1, 'p@5': -1, 'r@5': -1}
        else:
            return {'acc': -1, 'pre': -1, 'rec': -1, 'f1': -1, 'auc': -1}


class NCSparseModel(BaseModel):
    # Base Model for Node Classification Task

    def __init__(self, args):
        super(NCSparseModel, self).__init__(args)
        self.n_classes = args.n_classes
        self.decoder = model2decoder[args.model](args)
        # Calculate Weight Matrix to balance samples
        data = args.data['y']
        self.weights = self.get_weights(data)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def get_weights(self, data):
        alpha_pos = []
        alpha_neg = []
        for i in range(data.shape[1]):
            num_pos = torch.sum(data.to_dense().long()[:, i] == 1).float()
            num_neg = torch.sum(data.to_dense().long()[:, i] == 0).float()
            num_total = num_pos + num_neg
            alpha_pos.append(num_neg / num_total)
            alpha_neg.append(num_pos / num_total)
        return [alpha_pos, alpha_neg]

    def decode(self, h, adj):
        output = self.decoder.decode(h, adj)
        return output

    def get_loss(self, outputs, data, split):
        idx = data[f'idx_{split}']
        outputs = outputs
        labels = data['y']
        losses = [F.binary_cross_entropy_with_logits
                  (outputs[i], labels[i].to_dense().float(),
                   self.weights[0]*(labels[i].to_dense().long() == 1).float()
                   + self.weights[1]*(labels[i].to_dense().long() == 1).float()) / len(idx)
                  for i in idx]
        loss = losses[0]
        for i in range(1, len(idx)):
            loss = loss+losses[i]
        return loss

    def compute_metrics(self, outputs, data, split):
        outputs = torch.sigmoid(outputs)
        if outputs.is_cuda:
            outputs = outputs.detach().cpu()
            labels = data['y'].detach().cpu()
        idx = data[f'idx_{split}']
        outputs = outputs[idx]
        labels = labels[idx]
        f1_micro, f1_macro, auc_micro, auc_macro, p5, r5 = nc_metrics(outputs, labels)
        metrics = {'f1_micro': f1_micro, 'f1_macro': f1_macro,
                   'auc_micro': auc_micro, 'auc_macro': auc_macro, 'p@5': p5, 'r@5': r5}

        return metrics

    def has_improved(self, m1, m2):
        return m1['auc_macro'] < m2['auc_macro']

    def init_metric_dict(self):
        return {'f1_micro': -1, 'f1_macro': -1,
                'auc_micro': -1, 'auc_macro': -1, 'p@5': -1, 'r@5': -1}


class MultitaskNCModel1(BaseModel):
    # Model for Multi-task Node Classification Task

    def __init__(self, args):
        super(MultitaskNCModel1, self).__init__(args)
        self.dis_id = args.data['dis_id']
        self.med_id = args.data['med_id']
        self.dur_id = args.data['dur_id']
        args.n_classes = 50
        self.decoder_dis = model2decoder[args.model](args)
        self.decoder_med = model2decoder[args.model](args)
        args.n_classes = 1
        self.decoder_dur = model2decoder[args.model](args)
        # Calculate Weight Matrix to balance samples
        dis_y = args.data['dis_y']
        self.weights_dis = self.get_weights(dis_y)
        med_y = args.data['med_y']
        self.weights_med = self.get_weights(med_y)
        dur_y = args.data['dur_y']
        self.weights_dur = self.get_weights(dur_y)
        if not args.cuda == -1:
            self.weights_dis = self.weights_dis.to(args.device)
            self.weights_med = self.weights_med.to(args.device)
            self.weights_dur = self.weights_dur.to(args.device)

    def get_weights(self, data):
        pos = (data.long() == 1).float()
        neg = (data.long() == 0).float()
        alpha_pos = []
        alpha_neg = []
        if len(data.shape) > 1:
            for i in range(data.shape[1]):
                num_pos = torch.sum(data.long()[:, i] == 1).float()
                num_neg = torch.sum(data.long()[:, i] == 0).float()
                num_total = num_pos + num_neg
                alpha_pos.append(num_neg / num_total)
                alpha_neg.append(num_pos / num_total)
        else:
            num_pos = torch.sum(data.long() == 1).float()
            num_neg = torch.sum(data.long() == 0).float()
            num_total = num_pos + num_neg
            alpha_pos = num_neg / num_total
            alpha_neg = num_pos / num_total
        alpha_pos = torch.Tensor([alpha_pos] * data.shape[0])
        alpha_neg = torch.Tensor([alpha_neg] * data.shape[0])
        return alpha_pos * pos + alpha_neg * neg

    def decode(self, h, adj):
        output_dis = self.decoder_dis.decode(h, adj)
        output_med = self.decoder_med.decode(h, adj)
        output_dur = self.decoder_dur.decode(h, adj)

        return [output_dis, output_med, output_dur]

    def get_loss(self, outputs, data, split):
        outputs_dis, outputs_med, outputs_dur = tuple(outputs)

        dis_split = data[f'dis_{split}']
        dis_outputs = outputs_dis[self.dis_id][dis_split]
        dis_labels = data['dis_y'][dis_split]
        loss_dis = F.binary_cross_entropy_with_logits(dis_outputs, dis_labels.float(), self.weights_dis[dis_split])

        med_split = data[f'med_{split}']
        med_outputs = outputs_med[self.med_id][med_split]
        med_labels = data['med_y'][med_split]
        loss_med = F.binary_cross_entropy_with_logits(med_outputs, med_labels.float(), self.weights_med[med_split])

        dur_split = data[f'dur_{split}']
        dur_outputs = outputs_dur[self.dur_id][dur_split][:, 0]
        dur_labels = data['dur_y'][dur_split]
        loss_dur = F.binary_cross_entropy_with_logits(dur_outputs, dur_labels.float(), self.weights_dur[dur_split])
        return 0.8 * loss_dis + 0.15 * loss_med + 0.05 * loss_dur

    def compute_metrics(self, outputs, data, split):
        outputs_dis, outputs_med, outputs_dur = tuple(outputs)
        outputs_dis, outputs_med, outputs_dur = torch.sigmoid(outputs_dis), torch.sigmoid(outputs_med), torch.sigmoid(outputs_dur)

        dis_split = data[f'dis_{split}']
        if outputs_dis.is_cuda:
            outputs_dis = outputs_dis.detach().cpu()
            labels_dis = data['dis_y'].detach().cpu()
        dis_outputs = outputs_dis[self.dis_id][dis_split]
        dis_labels = labels_dis[dis_split]
        f1_micro_dis, f1_macro_dis, auc_micro_dis, auc_macro_dis, p5_dis, r5_dis = \
            nc_metrics(dis_outputs, dis_labels)

        med_split = data[f'med_{split}']
        if outputs_med.is_cuda:
            outputs_med = outputs_med.detach().cpu()
            labels_med = data['med_y'].detach().cpu()
        med_outputs = outputs_med[self.med_id][med_split]
        med_labels = labels_med[med_split]
        f1_micro_med, f1_macro_med, auc_micro_med, auc_macro_med, p5_med, r5_med = \
            nc_metrics(med_outputs, med_labels)

        dur_split = data[f'dur_{split}']
        if outputs_dur.is_cuda:
            outputs_dur = outputs_dur.detach().cpu()
            labels_dur = data['dur_y'].detach().cpu()
        dur_outputs = outputs_dur[self.dur_id][dur_split][:, 0]
        dur_labels = labels_dur[dur_split]
        acc, pre, rec, f1, auc = acc_f1(dur_outputs, dur_labels)

        metrics = {'f1_micro_dis': f1_micro_dis, 'f1_macro_dis': f1_macro_dis,
                   'auc_micro_dis': auc_micro_dis, 'auc_macro_dis': auc_macro_dis,
                   'p@5_dis': p5_dis, 'r@5_dis': r5_dis,
                   'f1_micro_med': f1_micro_med, 'f1_macro_med': f1_macro_med,
                   'auc_micro_med': auc_micro_med, 'auc_macro_med': auc_macro_med,
                   'p@5_med': p5_med, 'r@5_med': r5_med,
                   'acc_dur': acc, 'pre_dur': pre, 'rec_dur': rec, 'f1_dur': f1, 'auc_dur': auc}
        return metrics

    def has_improved(self, m1, m2):
        return m1['auc_macro_dis'] < m2['auc_macro_dis']

    def init_metric_dict(self):
        return {'f1_micro_dis': -1, 'f1_macro_dis': -1,
                'auc_micro_dis': -1, 'auc_macro_dis': -1,
                'p@5_dis': -1, 'r@5_dis': -1,
                'f1_micro_med': -1, 'f1_macro_med': -1,
                'auc_micro_med': -1, 'auc_macro_med': -1,
                'p@5_med': -1, 'r@5_med': -1,
                'acc_dur': -1, 'pre_dur': -1, 'rec_dur': -1, 'f1_dur': -1, 'auc_dur': -1}


class MultitaskNCModel2(BaseModel):
    # Model for Multi-task Node Classification Task

    def __init__(self, args):
        super(MultitaskNCModel2, self).__init__(args)
        self.dis_id = args.data['dis_id']
        self.med_id = args.data['med_id']
        self.dur_id = args.data['dur_id']
        self.encoder_dis = model2encoder[args.model](args)
        self.encoder_med = model2encoder[args.model](args)
        self.encoder_dur = model2encoder[args.model](args)
        args.n_classes = 50
        self.decoder_dis = model2decoder[args.model](args)
        self.decoder_med = model2decoder[args.model](args)
        args.n_classes = 1
        self.decoder_dur = model2decoder[args.model](args)
        # Calculate Weight Matrix to balance samples
        dis_y = args.data['dis_y']
        self.weights_dis = self.get_weights(dis_y)
        med_y = args.data['med_y']
        self.weights_med = self.get_weights(med_y)
        dur_y = args.data['dur_y']
        self.weights_dur = self.get_weights(dur_y)
        if not args.cuda == -1:
            self.weights_dis = self.weights_dis.to(args.device)
            self.weights_med = self.weights_med.to(args.device)
            self.weights_dur = self.weights_dur.to(args.device)
        d = args.dim
        init_range = np.sqrt(6.0 / (d + d))
        self.kernel_gate_dis = torch.FloatTensor(d, d).uniform_(-init_range, init_range)
        self.bias_gate_dis = torch.zeros([d])
        self.kernel_gate_med = torch.FloatTensor(d, d).uniform_(-init_range, init_range)
        self.bias_gate_med = torch.zeros([d])
        self.kernel_gate_dur = torch.FloatTensor(d, d).uniform_(-init_range, init_range)
        self.bias_gate_dur = torch.zeros([d])
        if not args.cuda == -1:
            self.kernel_gate_dis = self.kernel_gate_dis.to(args.device)
            self.bias_gate_dis = self.bias_gate_dis.to(args.device)
            self.kernel_gate_med = self.kernel_gate_med.to(args.device)
            self.bias_gate_med = self.bias_gate_med.to(args.device)
            self.kernel_gate_dur = self.kernel_gate_dur.to(args.device)
            self.bias_gate_dur = self.bias_gate_dur.to(args.device)
        x = args.data['x'].cpu().to_dense()
        x_dis = torch.zeros(x.shape[0], x.shape[1])
        x_med = torch.zeros(x.shape[0], x.shape[1])
        x_dur = torch.zeros(x.shape[0], x.shape[1])
        x_dis[self.dis_id] = x[self.dis_id]
        x_med[self.med_id] = x[self.med_id]
        x_dur[self.dur_id] = x[self.dur_id]
        self.x_dis = x_dis.to_sparse()
        self.x_med = x_med.to_sparse()
        self.x_dur = x_dur.to_sparse()
        if not args.cuda == -1:
            self.x_dis = self.x_dis.to(args.device)
            self.x_med = self.x_med.to(args.device)
            self.x_dur = self.x_dur.to(args.device)

    def get_weights(self, data):
        pos = (data.long() == 1).float()
        neg = (data.long() == 0).float()
        alpha_pos = []
        alpha_neg = []
        if len(data.shape) > 1:
            for i in range(data.shape[1]):
                num_pos = torch.sum(data.long()[:, i] == 1).float()
                num_neg = torch.sum(data.long()[:, i] == 0).float()
                num_total = num_pos + num_neg
                alpha_pos.append(num_neg / num_total)
                alpha_neg.append(num_pos / num_total)
        else:
            num_pos = torch.sum(data.long() == 1).float()
            num_neg = torch.sum(data.long() == 0).float()
            num_total = num_pos + num_neg
            alpha_pos = num_neg / num_total
            alpha_neg = num_pos / num_total
        alpha_pos = torch.Tensor([alpha_pos] * data.shape[0])
        alpha_neg = torch.Tensor([alpha_neg] * data.shape[0])
        return alpha_pos * pos + alpha_neg * neg

    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        if h.is_sparse:
            h = h.to_dense()
        h_dis = self.encoder_dis.encode(self.x_dis, adj)
        h_med = self.encoder_med.encode(self.x_med, adj)
        h_dur = self.encoder_dur.encode(self.x_dur, adj)

        transform_gate = torch.spmm(h, self.kernel_gate_dis) + self.bias_gate_dis
        transform_gate = torch.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate
        h_dis = transform_gate * h_dis + carry_gate * h

        transform_gate = torch.spmm(h, self.kernel_gate_med) + self.bias_gate_med
        transform_gate = torch.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate
        h_med = transform_gate * h_med + carry_gate * h

        transform_gate = torch.spmm(h, self.kernel_gate_dur) + self.bias_gate_dur
        transform_gate = torch.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate
        h_dur = transform_gate * h_dur + carry_gate * h

        return [h_dis, h_med, h_dur]

    def decode(self, h, adj):
        h_dis, h_med, h_dur = tuple(h)

        output_dis = self.decoder_dis.decode(h_dis, adj)
        output_med = self.decoder_med.decode(h_med, adj)
        output_dur = self.decoder_dur.decode(h_dur, adj)

        return [output_dis, output_med, output_dur]

    def get_loss(self, outputs, data, split):
        outputs_dis, outputs_med, outputs_dur = tuple(outputs)

        dis_split = data[f'dis_{split}']
        dis_outputs = outputs_dis[self.dis_id][dis_split]
        dis_labels = data['dis_y'][dis_split]
        loss_dis = F.binary_cross_entropy_with_logits(dis_outputs, dis_labels.float(), self.weights_dis[dis_split])

        med_split = data[f'med_{split}']
        med_outputs = outputs_med[self.med_id][med_split]
        med_labels = data['med_y'][med_split]
        loss_med = F.binary_cross_entropy_with_logits(med_outputs, med_labels.float(), self.weights_med[med_split])

        dur_split = data[f'dur_{split}']
        dur_outputs = outputs_dur[self.dur_id][dur_split][:, 0]
        dur_labels = data['dur_y'][dur_split]
        loss_dur = F.binary_cross_entropy_with_logits(dur_outputs, dur_labels.float(), self.weights_dur[dur_split])
        return 0.8 * loss_dis + 0.15 * loss_med + 0.05 * loss_dur

    def compute_metrics(self, outputs, data, split):
        outputs_dis, outputs_med, outputs_dur = tuple(outputs)
        outputs_dis, outputs_med, outputs_dur = torch.sigmoid(outputs_dis), torch.sigmoid(outputs_med), torch.sigmoid(
            outputs_dur)

        dis_split = data[f'dis_{split}']
        if outputs_dis.is_cuda:
            outputs_dis = outputs_dis.detach().cpu()
            labels_dis = data['dis_y'].detach().cpu()
        dis_outputs = outputs_dis[self.dis_id][dis_split]
        dis_labels = labels_dis[dis_split]
        f1_micro_dis, f1_macro_dis, auc_micro_dis, auc_macro_dis, p5_dis, r5_dis = \
            nc_metrics(dis_outputs, dis_labels)

        med_split = data[f'med_{split}']
        if outputs_med.is_cuda:
            outputs_med = outputs_med.detach().cpu()
            labels_med = data['med_y'].detach().cpu()
        med_outputs = outputs_med[self.med_id][med_split]
        med_labels = labels_med[med_split]
        f1_micro_med, f1_macro_med, auc_micro_med, auc_macro_med, p5_med, r5_med = \
            nc_metrics(med_outputs, med_labels)

        dur_split = data[f'dur_{split}']
        if outputs_dur.is_cuda:
            outputs_dur = outputs_dur.detach().cpu()
            labels_dur = data['dur_y'].detach().cpu()
        dur_outputs = outputs_dur[self.dur_id][dur_split][:, 0]
        dur_labels = labels_dur[dur_split]
        acc, pre, rec, f1, auc = acc_f1(dur_outputs, dur_labels)

        metrics = {'f1_micro_dis': f1_micro_dis, 'f1_macro_dis': f1_macro_dis,
                   'auc_micro_dis': auc_micro_dis, 'auc_macro_dis': auc_macro_dis,
                   'p@5_dis': p5_dis, 'r@5_dis': r5_dis,
                   'f1_micro_med': f1_micro_med, 'f1_macro_med': f1_macro_med,
                   'auc_micro_med': auc_micro_med, 'auc_macro_med': auc_macro_med,
                   'p@5_med': p5_med, 'r@5_med': r5_med,
                   'acc_dur': acc, 'pre_dur': pre, 'rec_dur': rec, 'f1_dur': f1, 'auc_dur': auc}
        return metrics

    def has_improved(self, m1, m2):
        return m1['auc_macro_dis'] < m2['auc_macro_dis']

    def init_metric_dict(self):
        return {'f1_micro_dis': -1, 'f1_macro_dis': -1,
                'auc_micro_dis': -1, 'auc_macro_dis': -1,
                'p@5_dis': -1, 'r@5_dis': -1,
                'f1_micro_med': -1, 'f1_macro_med': -1,
                'auc_micro_med': -1, 'auc_macro_med': -1,
                'p@5_med': -1, 'r@5_med': -1,
                'acc_dur': -1, 'pre_dur': -1, 'rec_dura': -1, 'f1_dur': -1, 'auc_dur': -1}
