import time

import scipy
import ot
from utils.data_utils import *
from utils.eval_utils import format_metrics, eval_gw_matching_matrix, eval_at_1
from models.models_ea import EAModel, UEAModel
from SinkhornOT.iterative_projection import gw_iterative_1

def get_cost_matrix(M, device):
    C = 1 / (M + 1)
    C[M == 0] = 1
    return C.to(device)


def train_gw_ea(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    print(f'Using: {args.device}')
    print(f'Using seed: {args.seed}')

    # Load Data
    data = load_seperate_data_ea(args)
    M1, M2 = data['adj1'], data['adj2']

    n, m = len(M1), len(M2)
    M1, M2 = M1.to_dense(), M2.to_dense()
    mu, nu = torch.ones(n).to(args.device) / n, torch.ones(m).to(args.device) / m
    # mu, nu = torch.ones(n).to(args.device), torch.ones(m).to(args.device)
    C1, C2 = get_cost_matrix(M1, args.device), get_cost_matrix(M2, args.device)
    import ot
    T, log = ot.gromov.entropic_gromov_wasserstein(C1.cpu().numpy(), C2.cpu().numpy(), mu.cpu().numpy(), nu.cpu().numpy(), loss_fun="square_loss", epsilon=1e-5, verbose=True, log=True)
    T = torch.from_numpy(T).to(args.device)
    # T, log = gw_iterative_1(C1, C2, mu, nu, epsilon=0.0001, max_iter=10, log=True, tol=1e-9)
    # torch.save(T, "output/T.plt")   
    # torch.save(log, "output/log.plt")   

    metrics = eval_gw_matching_matrix(-T, data["test"][:n], data["index1_R"], data["index2_R"])
    print(format_metrics(metrics, "gw"))


def train_unsup_ea(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    print(f'Using: {args.device}')
    print(f'Using seed: {args.seed}')

    # Load Data
    data = load_seperate_data_ea(args)
    args.n_nodes, args.feat_dim = data['x'].shape
    print(f'Num_nodes: {args.n_nodes}')
    print(f'Dim_feats: {args.feat_dim}')
    args.data = data
    Model = None
    args.n_classes = args.feat_dim

    Model = UEAModel
    # Model and Optimizer
    model = Model(args)
    print(str(model))
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    # Train Model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    print("Only Glove Hits@1: {:.4f}".format(eval_at_1(data['x'].to_dense(), data)))
    # Minimize Wassertein Distance
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        semantic = data['x']
        # semantic = torch.zeros_like(data['x']).to_dense().to(data['x'].device)
        # semantic[:, 1] = 1
        embeddings = model.encode(semantic, data['adj'])
        outputs = model.decode(embeddings, data['adj'])

        for it in range(args.iters):
            loss = model.get_loss_wassertein(outputs, data, args.batch_size)
            # loss = model.get_loss_gromove_wassertein(outputs, data, args.batch_size)
            loss.backward()
            optimizer.step()
            # embeddings = model.encode(data['x'], data['adj'])
            # outputs = model.decode(embeddings, data['adj'])

        lr_scheduler.step()
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(semantic, data['adj'])
            outputs = model.decode(embeddings, data['adj'])
            # outputs, outputs_r = model.encode(data['idx_x'], data['idx_r'])
            # metrics = model.compute_metrics(outputs, data, 'val')
            hits1 = eval_at_1(outputs, data)
            print(' '.join(['Epoch: {:04d}'.format(epoch + 1), 'Loss: {:.4f}'.format(loss),
                            'Test Hits@1: {:.4f}%'.format(hits1)]))
    
    # Refinement
    for epoch in range(args.refine_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(semantic, data['adj'])
        outputs = model.decode(embeddings, data['adj'])

        if epoch % 10 == 0:
            model.generate_pairs(outputs, data, args.batch_size)
            model.generate_neg(outputs, args.neg_num)

        loss = model.get_loss(outputs)
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        model.eval()
        embeddings = model.encode(semantic, data['adj'])
        outputs = model.decode(embeddings, data['adj'])
        # outputs, outputs_r = model.encode(data['idx_x'], data['idx_r'])
        hits1 = eval_at_1(outputs, data)
        print(' '.join(['Refine-Epoch: {:04d}'.format(epoch + 1), 'Loss: {:.4f}'.format(loss),
                        'Test Hits@1: {:.4f}%'.format(hits1)]))

    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['x'], data['adj'])
        outputs = model.decode(best_emb, data['adj'])
        best_test_metrics = model.compute_metrics(outputs, data, 'test')
    print(' '.join(['Val set results:',
                    format_metrics(best_val_metrics, 'val')]))
    print(' '.join(['Test set results:',
                    format_metrics(best_test_metrics, 'test')]))
    if args.save:
        np.save(f'data/dbp15k/{args.dataset}/{args.model}_embeddings.npy', best_emb.cpu().detach().numpy())
        args.data = []
        json.dump(vars(args), open(f'data/dbp15k/{args.dataset}/{args.model}_config.json', 'w'))
        torch.save(model.state_dict(), f'data/dbp15k/{args.dataset}/{args.model}_model.pth')
        print(f'Saved model!')


def train_bli_ea(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    print(f'Using: {args.device}')
    print(f'Using seed: {args.seed}')

    # Load Data
    data = load_seperate_data_ea(args)
    args.n_nodes, args.feat_dim = data['x'].shape
    print(f'Num_nodes: {args.n_nodes}')
    print(f'Dim_feats: {args.feat_dim}')
    args.data = data
    args.n_classes = args.feat_dim

    # Model and Optimizer
    if args.cuda is not None and int(args.cuda) >= 0:
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    Model = UEAModel
    model = Model(args)
    # Train Model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    print("Only Glove Hits@1: {:.4f}".format(eval_at_1(data['x'].to_dense(), data)))
    # Minimize Wassertein Distance

    embeddings = data['x'].to_dense()

    # normalize
    embeddings = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True) + 1e-8
    # center
    embeddings = embeddings - torch.mean(embeddings, dim=1, keepdim=True)
    # normalize
    embeddings = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True) + 1e-8

    e1, e2 = data['e1'], data['e2']
    index1, index2 = data['index1'], data['index2']
    L = np.array([index1[i] for i in np.arange(e1)])
    R = np.array([index2[i] for i in np.arange(e2)])
     
    def Procrutes(src_emb_gw, tgt_emb_gw, T):
        T = T.to('cpu') # scipy can only operate on cpu
        X = src_emb_gw.to('cpu')
        Y = T.mm(tgt_emb_gw.to('cpu')).to('cpu')
        M = X.t().mm(Y)
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        Q = torch.from_numpy(U.dot(V_t))
        return Q

    for epoch in range(args.epochs):
        t = time.time()
        X = embeddings[L][:5000, ]
        Y = embeddings[R][:5000, ]

        C1 = torch.cdist(X, X, p = 2).detach()
        C2 = torch.cdist(Y, Y, p = 2).detach()
        a, b = torch.ones(len(C1)).to(args.device), torch.ones(len(C2)).to(args.device)
        # T, gwdist = gw_iterative_1(C1, C2, a, b, epsilon=0.001, max_iter=1000)
        T = ot.gromov.entropic_gromov_wasserstein(C1.cpu().numpy(), C2.cpu().numpy(), a.cpu().numpy(), b.cpu().numpy(), 'square_loss', epsilon=5e-5)

        Q = Procrutes(X, Y, torch.from_numpy(T)).to(args.device)
        embeddings[L] = embeddings[L].mm(Q)
        loss = torch.norm(X.mm(Q) - T[0].mm(Y))

        if (epoch + 1) % args.eval_freq == 0:
            hits1 = eval_at_1(embeddings, data)
            print(' '.join(['Epoch: {:04d}'.format(epoch + 1), 'Loss: {:.4f}'.format(loss),
                            'Test Hits@1: {:.4f}%'.format(hits1)]))