from config import parser
from run.train_nc import train_nc
from run.train_ea import train_ea
from run.train_nctext import train_nctext
from run.train_ncfull import train_ncfull


if __name__ == '__main__':
    args = parser.parse_args()
    if args.task == 'nc':
        if args.dataset == 'full':
            train_ncfull(args)
        else:
            train_nc(args)
    elif args.task == 'ea':
        train_ea(args)
    elif args.task == 'nctext':
        train_nctext(args)
