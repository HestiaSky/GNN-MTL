import time

from utils.data_utils import *
from utils.eval_utils import format_metrics
from models.models_nctext import LogisticRegression, Multilayer, BidirectionalGRU, TextCNN, HAN


def train_nctext(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    print(f'Using: {args.device}')
    print(f'Using seed: {args.seed}')

    # Load Data
    data = load_data(args)
    if args.model not in ['bigru', 'textcnn', 'han']:
        args.n_nodes, args.feat_dim = data['x'].shape
        print(f'Num_nodes: {args.n_nodes}')
        print(f'Dim_feats: {args.feat_dim}')
    args.data = data
    if args.model == 'lr':
        Model = LogisticRegression
    elif args.model == 'mlp':
        Model = Multilayer
    elif args.model == 'bigru':
        Model = BidirectionalGRU
    elif args.model == 'textcnn':
        Model = TextCNN
    elif args.model == 'han':
        Model = HAN
    if args.dataset == 'dur':
        args.n_classes = 1
    else:
        args.n_classes = 50
    print(f'Num Labels: {args.n_classes}')

    # Model and Optimizer
    model = Model(args)
    print(str(model))
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    # Train Model
    t_total = time.time()
    counter = 0
    best_emb = None
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        if args.model in ['bigru', 'textcnn', 'han']:
            outputs = []
            labels = []
            for batch in data['train_iter']:
                if len(batch) != args.batch_size:
                    continue
                feature, target = batch.text, batch.label
                if args.model == 'han':
                    feature = han_data(feature)
                if args.dataset != 'dur':
                    target = target.transpose(0, 1).sub_(2)
                else:
                    target = target.view(-1, 1).sub_(1)
                if not args.cuda == -1:
                    feature, target = feature.to(args.device), target.to(args.device)
                optimizer.zero_grad()
                logits = model.forward(feature)
                outputs.append(logits)
                labels.append(target)
                loss = model.get_loss(logits, target, 'train')
                loss.backward()
                optimizer.step()
            outputs = torch.cat(outputs, 0)
            labels = torch.cat(labels, 0)
            data['y'] = labels
        else:
            optimizer.zero_grad()
            outputs = model.forward(data['x'])
            loss = model.get_loss(outputs, data, 'train')
            loss.backward()
            optimizer.step()
        if (epoch + 1) % args.log_freq == 0:
            train_metrics = model.compute_metrics(outputs, data, 'train')
            print(' '.join(['Epoch: {:04d}'.format(epoch + 1),
                            format_metrics(train_metrics, 'train'),
                            'time: {:.4f}s'.format(time.time() - t)]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            if args.model in ['bigru', 'textcnn', 'han']:
                outputs = []
                labels = []
                for batch in data['val_iter']:
                    if len(batch) != args.batch_size:
                        continue
                    feature, target = batch.text, batch.label
                    if args.dataset != 'dur':
                        target = target.transpose(0, 1).sub_(2)
                    else:
                        target = target.view(-1, 1).sub_(1)
                    if not args.cuda == -1:
                        feature, target = feature.to(args.device), target.to(args.device)
                    logits = model.forward(feature)
                    outputs.append(logits)
                    labels.append(target)
                outputs = torch.cat(outputs, 0)
                labels = torch.cat(labels, 0)
                data['y'] = labels
            else:
                outputs = model.forward(data['x'])
            val_metrics = model.compute_metrics(outputs, data, 'val')
            print(' '.join(['Epoch: {:04d}'.format(epoch + 1),
                            format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                if args.model in ['bigru', 'textcnn', 'han']:
                    outputs = []
                    labels = []
                    for batch in data['test_iter']:
                        if len(batch) != args.batch_size:
                            continue
                        feature, target = batch.text, batch.label
                        if args.dataset != 'dur':
                            target = target.transpose(0, 1).sub_(2)
                        else:
                            target = target.view(-1, 1).sub_(1)
                        if not args.cuda == -1:
                            feature, target = feature.to(args.device), target.to(args.device)
                        logits = model.forward(feature)
                        outputs.append(logits)
                        labels.append(target)
                    outputs = torch.cat(outputs, 0)
                    labels = torch.cat(labels, 0)
                    data['y'] = labels
                best_emb = outputs
                best_test_metrics = model.compute_metrics(outputs, data, 'test')
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
        if args.model in ['bigru', 'textcnn', 'han']:
            outputs = []
            labels = []
            for batch in data['test_iter']:
                if len(batch) != args.batch_size:
                    continue
                feature, target = batch.text, batch.label
                if args.dataset != 'dur':
                    target = target.transpose(0, 1).sub_(2)
                else:
                    target = target.view(-1, 1).sub_(1)
                if not args.cuda == -1:
                    feature, target = feature.to(args.device), target.to(args.device)
                logits = model.forward(feature)
                outputs.append(logits)
                labels.append(target)
            outputs = torch.cat(outputs, 0)
            labels = torch.cat(labels, 0)
            data['y'] = labels
        else:
            outputs = model.forward(data['x'])
        best_test_metrics = model.compute_metrics(outputs, data, 'test')
    print(' '.join(['Val set results:',
                    format_metrics(best_val_metrics, 'val')]))
    print(' '.join(['Test set results:',
                    format_metrics(best_test_metrics, 'test')]))

