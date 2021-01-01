from config import parser
from run.train_ea import train_ea
from run.train_unsup_ea import train_unsup_ea, train_bli_ea

if __name__ == '__main__':
    args = parser.parse_args()
    train_ea(args)
    # train_unsup_ea(args)
    # train_bli_ea(args)
 