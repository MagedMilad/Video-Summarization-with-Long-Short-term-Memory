import argparse, textwrap
from argparse import RawTextHelpFormatter
from preprocess import *
from test import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='vsLSTM', formatter_class=RawTextHelpFormatter)

    parser.add_argument('-bs', '--batch-size', dest='batch_size', default=16)
    parser.add_argument('-ts', '--time-step', dest='time_step', default=10)
    parser.add_argument('-ds', '--dataset', dest='dataset', default='t',choices=['s','t'],
                        help="Dataset:\n\tSumMe: s\n\tTVSum: t\n(default: %(default)s)")
    parser.add_argument('-st', '--setting', dest='setting', default='t',choices=['c','a','t'],
                        help='Settings:\n\tCanonical: c\n\tAugmented: a\n\tTransfer : t\n(default: %(default)s)')
    parser.add_argument('-sh', '--shuffle', dest='shuffle', default=True, type=bool,
                        help='shuffle the data (default: %(default)s)')

    args = parser.parse_args()

    load_settings(args.dataset, args.setting, args.shuffle)
    train(args.batch_size, args.dataset, args.setting, args.time_step)


    # print args
