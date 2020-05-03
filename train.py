from config import parser
from run.train_nc import train_nc
from run.train_ea import train_ea
from run.train_nctext import train_nctext


if __name__ == '__main__':
    args = parser.parse_args()
    if args.task == 'nc':
        train_nc(args)
    elif args.task == 'ea':
        train_ea(args)
    elif args.task == 'nctext':
        train_nctext(args)
