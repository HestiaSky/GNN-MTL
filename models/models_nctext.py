import torch
from torch import nn
import torch.nn.functional as F
from utils.eval_utils import *
from torch.autograd import Variable


class BaseModel(nn.Module):
    # Base Model for KG Embedding Task

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.n_classes = args.n_classes
        self.n_dim = args.dim
        self.layer = None
        if args.model not in ['bigru', 'textcnn', 'han']:
            self.weights = self.get_weights(args.data['y'])
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

    def forward(self, x):
        output = self.layer.forward(x)
        return output

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
        if self.n_classes > 1:
            f1_micro, f1_macro, auc_micro, auc_macro, p5, r5 = nc_metrics(outputs, labels, self.n_classes)
            metrics = {'f1_micro': f1_micro, 'f1_macro': f1_macro,
                       'auc_micro': auc_micro, 'auc_macro': auc_macro, 'p@5': p5, 'r@5': r5}
        else:
            acc, pre, rec, f1 = acc_f1(outputs, labels)
            metrics = {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1}
        return metrics

    def has_improved(self, m1, m2):
        if self.n_classes > 1:
            return m1['auc_macro'] < m2['auc_macro']
        else:
            return m1['acc'] < m2['acc']

    def init_metric_dict(self):
        if self.n_classes > 1:
            return {'f1_micro': -1, 'f1_macro': -1,
                    'auc_micro': -1, 'auc_macro': -1, 'p@5': -1, 'r@5': -1}
        else:
            return {'acc': -1, 'pre': -1, 'rec': -1, 'f1': -1}


class LogisticRegression(BaseModel):
    # Logistic Regression Model

    def __init__(self, args):
        super(LogisticRegression, self).__init__(args)
        self.layer = nn.Linear(self.n_dim, self.n_classes, True)


class Multilayer(BaseModel):
    # Multilayer Model

    def __init__(self, args):
        super(Multilayer, self).__init__(args)
        layers = []
        layers.append(nn.Linear(self.n_dim, self.n_dim, True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.n_dim, self.n_dim, True))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.n_dim, self.n_classes, True))
        self.layer = nn.Sequential(*layers)


class BidirectionalGRU(BaseModel):
    # Bidirectional GRU Model

    def __init__(self, args):
        super(BidirectionalGRU, self).__init__(args)
        self.hidden_size = self.n_dim
        self.num_layers = 2
        vocab = args.data['TEXT'].vocab
        self.embed = nn.Embedding(len(vocab), self.n_dim)
        self.bigru = nn.GRU(self.n_dim, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * 2, self.n_classes, True)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout(embed)
        x = embed.view(len(x), embed.size(1), -1)
        # Forward propagate LSTM
        gru_out, _ = self.bigru(x)
        gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        # Decode the hidden state of the last time step
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = F.tanh(gru_out)
        out = self.fc(gru_out)
        return out

    def get_loss(self, logits, targets, split):
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        return loss

    def compute_metrics(self, outputs, data, split):
        labels = data['y']
        if self.n_classes > 1:
            f1_micro, f1_macro, auc_micro, auc_macro, p5, r5 = nc_metrics(outputs, labels, self.n_classes)
            metrics = {'f1_micro': f1_micro, 'f1_macro': f1_macro,
                       'auc_micro': auc_micro, 'auc_macro': auc_macro, 'p@5': p5, 'r@5': r5}
        else:
            acc, pre, rec, f1 = acc_f1(outputs, labels)
            metrics = {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1}
        return metrics


class TextCNN(BaseModel):
    # Text CNN Model

    def __init__(self, args):
        super(TextCNN, self).__init__(args)
        vocab = args.data['TEXT'].vocab
        self.embed = nn.Embedding(len(vocab), self.n_dim)
        self.conv3 = nn.Conv2d(1, 50, (3, self.n_dim))
        self.conv4 = nn.Conv2d(1, 50, (4, self.n_dim))
        self.conv5 = nn.Conv2d(1, 50, (5, self.n_dim))
        self.fc = nn.Linear(3*50, self.n_classes, True)

    def forward(self, x):
        embed = self.embed(x)
        x = embed.view(len(x), embed.size(1), -1)
        x = torch.transpose(x, 0, 1)
        x = x.view(x.shape[0], 1, x.shape[1], -1)
        batch = x.shape[0]
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))
        # Pooling
        x1 = x1.view(batch, 50, -1)
        x2 = x2.view(batch, 50, -1)
        x3 = x3.view(batch, 50, -1)
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        # project the features to the labels
        out = self.fc(x)
        return out

    def get_loss(self, logits, targets, split):
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        return loss

    def compute_metrics(self, outputs, data, split):
        labels = data['y']
        if self.n_classes > 1:
            f1_micro, f1_macro, auc_micro, auc_macro, p5, r5 = nc_metrics(outputs, labels, self.n_classes)
            metrics = {'f1_micro': f1_micro, 'f1_macro': f1_macro,
                       'auc_micro': auc_micro, 'auc_macro': auc_macro, 'p@5': p5, 'r@5': r5}
        else:
            acc, pre, rec, f1 = acc_f1(outputs, labels)
            metrics = {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1}
        return metrics


def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
        if nonlinearity =='tanh':
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if s is None:
            s = _s_bias
        else:
            s = torch.cat((s, _s_bias), 0)
    return s.squeeze()


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if nonlinearity == 'tanh':
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if s is None:
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze()


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if attn_vectors is None:
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


class AttentionWordRNN(nn.Module):
    def __init__(self, batch_size, num_tokens, embed_size, word_gru_hidden, bidirectional=True):
        super(AttentionWordRNN, self).__init__()

        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        self.lookup = nn.Embedding(num_tokens, embed_size)

        if bidirectional:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=True)
            self.weight_W_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 2 * word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2 * word_gru_hidden, 1))
        else:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional=False)
            self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden, word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))

        self.softmax_word = nn.Softmax(dim=1)
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1, 0.1)

    def forward(self, embed, state_word):
        # embeddings
        embedded = self.lookup(embed)
        # word level gru
        output_word, state_word = self.word_gru(embedded, state_word)
        #         print output_word.size()
        word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn.transpose(1, 0))
        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1, 0))
        return word_attn_vectors, state_word, word_attn_norm

    def init_hidden(self):
        if self.bidirectional:
            return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))


class AttentionSentRNN(nn.Module):

    def __init__(self, batch_size, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional=True):

        super(AttentionSentRNN, self).__init__()

        self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional

        if bidirectional:
            self.sent_gru = nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional=True)
            self.weight_W_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 2 * sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(2 * sent_gru_hidden, 1))
            self.final_linear = nn.Linear(2 * sent_gru_hidden, n_classes)
        else:
            self.sent_gru = nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional=True)
            self.weight_W_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, sent_gru_hidden))
            self.bias_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
            self.weight_proj_sent = nn.Parameter(torch.Tensor(sent_gru_hidden, 1))
            self.final_linear = nn.Linear(sent_gru_hidden, n_classes)
        self.softmax_sent = nn.Softmax(dim=1)
        self.weight_W_sent.data.uniform_(-0.1, 0.1)
        self.weight_proj_sent.data.uniform_(-0.1, 0.1)

    def forward(self, word_attention_vectors, state_sent):
        output_sent, state_sent = self.sent_gru(word_attention_vectors, state_sent)
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent, self.bias_sent, nonlinearity='tanh')
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)
        sent_attn_norm = self.softmax_sent(sent_attn.transpose(1, 0))
        sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1, 0))
        # final classifier
        final_map = self.final_linear(sent_attn_vectors.squeeze(0))
        return final_map, state_sent, sent_attn_norm

    def init_hidden(self):
        if self.bidirectional:
            return Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.sent_gru_hidden))


class HAN(BaseModel):
    # Hieriarchical Attention Network Model

    def __init__(self, args):
        super(HAN, self).__init__(args)
        vocab = args.data['TEXT'].vocab
        self.batch_size = args.batch_size
        self.word_attn = AttentionWordRNN(batch_size=args.batch_size, num_tokens=len(vocab),
                                          embed_size=self.n_dim, word_gru_hidden=self.n_dim, bidirectional=True)
        self.sent_attn = AttentionSentRNN(batch_size=args.batch_size, sent_gru_hidden=self.n_dim,
                                          word_gru_hidden=self.n_dim, n_classes=self.n_classes, bidirectional=True)
        self.state_word = self.word_attn.init_hidden()
        self.state_sent = self.sent_attn.init_hidden()
        if not args.cuda == -1:
            self.state_word = self.state_word.to(args.device)
            self.state_sent = self.state_sent.to(args.device)

    def forward(self, x):
        print(x.shape)
        max_sents, batch_size, max_tokens = x.size()
        s = None
        for i in range(max_sents):
            _s, state_word, _ = self.word_attn(x[i, :, :].transpose(0, 1), self.state_word)
            if s is None:
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        y_pred, state_sent, _ = self.sent_attn(s, self.state_sent)
        return y_pred

    def get_loss(self, logits, targets, split):
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        return loss

    def compute_metrics(self, outputs, data, split):
        labels = data['y']
        if self.n_classes > 1:
            f1_micro, f1_macro, auc_micro, auc_macro, p5, r5 = nc_metrics(outputs, labels, self.n_classes)
            metrics = {'f1_micro': f1_micro, 'f1_macro': f1_macro,
                       'auc_micro': auc_micro, 'auc_macro': auc_macro, 'p@5': p5, 'r@5': r5}
        else:
            acc, pre, rec, f1 = acc_f1(outputs, labels)
            metrics = {'acc': acc, 'pre': pre, 'rec': rec, 'f1': f1}
        return metrics
