import argparse


config_args = {
    'lr': 0.002,
    'dropout': 0.2,
    'cuda': 1,
    'epochs': 10000,
    'weight-decay': 0.0,
    'seed': 10086,
    'task': 'nc',
    'model': 'HGCN',
    'num-layers': 3,
    'act': 'relu',
    'dim': 200,
    'n-heads': 4,
    'alpha': 0.2,
    'dataset': 'med',
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
    'save': 0
}

parser = argparse.ArgumentParser()
for param, val in config_args.items():
    parser.add_argument(f"--{param}", action="append", default=val)
