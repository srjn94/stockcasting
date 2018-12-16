import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='data')
parser.add_argument('-t', '--train_range', nargs=2, type=str, default=['2017-05-01','2018-08-31'])
parser.add_argument('-v', '--dev_range', nargs=2, type=str, default=['2018-10-01','2018-11-30'])

if __name__ == "__main__":
    args = parser.parse_args()
    args.train_range = [datetime.strptime(x, "%Y-%m-%d") for x in args.train_range]
    args.dev_range = [datetime.strptime(x, "%Y-%m-%d") for x in args.dev_range]

    path_dates = os.path.join(args.data_dir, "dates.txt")
    with open(path_dates, "r") as f_dates:
        datestrings = [datestring.strip() for datestring in f_dates]

    path_signals = os.path.join(args.data_dir, "signals")
    for root, dirs, files in os.walk(path_signals):
        if not files:
            continue
        train_root = root.replace("signals", "train")
        dev_root = root.replace("signals", "dev")
        for new_root in [train_root, dev_root]:
            if not os.path.isdir(new_root):
                os.makedirs(new_root)
        for name in files:
            with open(os.path.join(root, name), "r") as fin:
                lines = [line.strip() for line in fin]
            with open(os.path.join(train_root, name), "w") as ftrain:
                with open(os.path.join(dev_root, name), "w") as fdev:
                    for i, line in enumerate(lines):
                        datestamp = datetime.strptime(datestrings[i], "%Y-%m-%d")
                        if args.train_range[0] <= datestamp and datestamp <= args.train_range[1]:
                            ftrain.write(line + "\n")
                        if args.dev_range[0] <= datestamp and datestamp <= args.dev_range[1]:
                            fdev.write(line + "\n")
        with open(os.path.join(train_root, "dates.txt"), "w") as ftrain_dates:
            with open(os.path.join(dev_root, "dates.txt"), "w") as fdev_dates:
                for datestring in datestrings:
                    datestamp = datetime.strptime(datestring, "%Y-%m-%d")
                    if args.train_range[0] <= datestamp and datestamp <= args.train_range[1]:
                        ftrain_dates.write(datestring + "\n")
                    if args.dev_range[0] <= datestamp and datestamp <= args.dev_range[1]:
                        fdev_dates.write(datestring + "\n")
