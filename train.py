from config import parser
import torch
import torch.nn.functional as F
import numpy as np
from utils import load_data, preprocess_adj 
from models import GCN_ALIGN
import time

args = parser.parse_args()

# Load Data
adj, ae_input, train, test = load_data(args)
e = adj.shape[0]
se_input = torch.sparse.FloatTensor(torch.from_numpy(np.array([list(range(e)), list(range(e))]).astype(np.int64)), torch.ones(e), (e, e))
print("Data Loaded! Entity Num: {:4d}, Atrribute Dim: {:4d}, Train Size: {:4d}, Test Size: {:4d}".format(e, ae_input.shape[1], len(train), len(test)))

# Adj Prprocessing
adj = preprocess_adj(adj) 

# Model and Optimizer
se_model = GCN_ALIGN(neg_k = args.neg_k, in_features=se_input.shape[1], out_features=args.se_dim, dropout=args.dropout, act=getattr(F, args.act), sparse_feature=True, use_bias=False)
optimizer = torch.optim.SGD(params=se_model.parameters(), lr=args.lr)
tot_params = sum([np.prod(p.size()) for p in se_model.parameters()])
print(f'Total number of parameters: {tot_params}')

# Set seed and cuda
np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
print(f'Using: {args.device}')
print(f'Using seed: {args.seed}')
if args.cuda is not None and int(args.cuda) >= 0:
    se_model = se_model.to(args.device)
    adj = adj.to(args.device)
    ae_input = ae_input.to(args.device)
    se_input = se_input.to(args.device)

# Train model
for epoch in range(args.epoches):
    if not epoch % 10:
        se_model.sample_neg(train, e)
    s = time.time()
    optimizer.zero_grad()
    loss = se_model.get_loss(se_input, adj, train, args.gamma)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % args.eval_freq == 0:
        hits1 = se_model.eval_at_1(test)
        print(' '.join(['Epoch: {:04d}'.format(epoch + 1), 'Loss: {:.4f}'.format(loss),
                        'Test Hits@1: {:.4f}%'.format(hits1)]))
    else:
        print(' '.join(['Epoch: {:04d}'.format(epoch + 1), 'Loss: {:.4f}'.format(loss)]))
