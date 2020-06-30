import argparse


config_args = {
    'lr': 0.001,
    'dropout': 0.0,
    'cuda': -1,
    'epochs': 10000,
    'weight-decay': 0.0,
    'seed': 10086,
    'task': 'nctext',
    'model': 'textcnn',
    'num-layers': 3,
    'act': 'relu',
    'dim': 300,
    'n-heads': 4,
    'alpha': 0.2,
    'dataset': 'dis',
    'normalize_x': 0,
    'normalize_adj': 1,
    'patience': 50,
    'log-freq': 10,
    'eval-freq': 10,
    'lr-reduce-freq': 2000,
    'gamma': 0.5,
    'min-epochs': 100,
    'use_feats': 1,
    'bias': 1,
    'neg_num': 125,
    'batch_size': 128,
    'save': 0
}

parser = argparse.ArgumentParser()
for param, val in config_args.items():
    parser.add_argument(f"--{param}", action="append", default=val)
