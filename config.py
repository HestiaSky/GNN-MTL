import argparse


config_args = {
    'lr': 0.01,
    'dropout': 0.0,
    'cuda': -1,
    'epochs': 5000,
    'weight-decay': 0.0,
    'seed': 10086,
    'task': 'nc',
    'model': 'GAT',
    'num-layers': 2,
    'act': 'relu',
    'dim': 100,
    'n-heads': 4,
    'alpha': 0.2,
    'dataset': 'disc',
    'normalize_x': 1,
    'normalize_adj': 1,
    'patience': 100,
    'log-freq': 1,
    'eval-freq': 1,
    'lr-reduce-freq': 50,
    'gamma': 0.1,
    'min-epochs': 100,
    'use_feats': 0
}

parser = argparse.ArgumentParser()
for param, val in config_args.items():
    parser.add_argument(f"--{param}", action="append", default=val)
