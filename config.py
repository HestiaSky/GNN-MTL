import argparse

config_args = {
    'seed': 10086,
    'cuda': 1,
    'lang': 'zh_en',
    'split': 7,
    'act': 'relu',
    'dropout': 0.,
    'se_dim': 200,
    'lr': 20,
    'epoches': 2000,
    'neg_k': 5,
    'eval_freq': 100,
    'gamma': 3.0
}

parser = argparse.ArgumentParser()
for param, val in config_args.items():
    parser.add_argument(f"--{param}", action="append", default=val)
