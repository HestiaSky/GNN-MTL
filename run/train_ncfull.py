import time

from utils.data_utils import *
from utils.eval_utils import format_metrics
from models.models_nc import NCSparseModel


def train_ncfull(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    print(f'Using: {args.device}')
    print(f'Using seed: {args.seed}')

    # Load Data
    data = load_data(args)
    args.n_nodes, args.feat_dim = data['x'].shape
    print(f'Num_nodes: {args.n_nodes}')
    print(f'Dim_feats: {args.feat_dim}')
    args.data = data
    Model = None
    if args.task == 'nc' and args.dataset == 'full':
        Model = NCSparseModel
        args.n_classes = data['y'].shape[1]
        print(f'Num Labels: {args.n_classes}')

    # Model and Optimizer
    model = Model(args)
    print(str(model))
    optimizer_encoder = torch.optim.Adam(params=model.encoder.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    optimizer_decoder = torch.optim.Adam(params=model.decoder.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]) and x != 'y':
                data[x] = data[x].to(args.device)

    # Train Model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer_encoder.zero_grad()
        embeddings = model.encode(data['x'], data['adj'])
        embeddings = torch.cat([embeddings, data['x'].to_dense()], axis=1)
        batch_num = len(data['batch'])
        for step, (batch_x, batch_y) in enumerate(data['batch']):
            if args.cuda is not None and int(args.cuda) >= 0:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            optimizer_decoder.zero_grad()
            outputs = model.decode(embeddings[batch_x], data['adj'])
            loss = model.get_loss(outputs, batch_y, 'train')
            if (step + 1) < batch_num:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizer_decoder.step()
        optimizer_encoder.step()
        if (epoch + 1) % args.log_freq == 0:
            train_metrics = model.compute_metrics(outputs, data, 'train')
            print(' '.join(['Epoch: {:04d}'.format(epoch + 1),
                            'lr: {}'.format(args.lr),
                            format_metrics(train_metrics, 'train'),
                            'time: {:.4f}s'.format(time.time() - t)]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['x'], data['adj'])
            embeddings = torch.cat([embeddings, data['x'].to_dense()], axis=1)
            outputs = model.decode(embeddings, data['adj'])
            val_metrics = model.compute_metrics(outputs, data, 'val')
            print(' '.join(['Epoch: {:04d}'.format(epoch + 1),
                            format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(outputs, data, 'test')
                '''if type(embeddings) == type([]):
                    best_emb = [x.cpu() for x in embeddings]
                else:
                    best_emb = x.cpu()
                if args.save:
                    np.save('embeddings.npy', best_emb.detach().numpy())'''
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    print("Early stopping")
                    break

    print('Optimization Finished!')
    print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['x'], data['adj'])
        best_emb = torch.cat([best_emb, data['x'].to_dense()], axis=1)
        outputs = model.decode(best_emb, data['adj'])
        best_test_metrics = model.compute_metrics(outputs, data, 'test')
    print(' '.join(['Val set results:',
                    format_metrics(best_val_metrics, 'val')]))
    print(' '.join(['Test set results:',
                    format_metrics(best_test_metrics, 'test')]))
    if args.save:
        np.save('embeddings.npy', best_emb.cpu().detach().numpy())
        json.dump(vars(args), open('config.json', 'w'))
        torch.save(model.state_dict(), 'model.pth')
        print(f'Saved model!')

