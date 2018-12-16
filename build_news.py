import argparse
import numpy as np
import os
import re
import pandas as pd
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_companies", type=int, default=50)
parser.add_argument("-d", "--data_dir", type=str, default="data")
parser.add_argument("-f", "--stocks_file", type=str, default="data/companylist.csv")

def get_keyphrases(symbol, name):
    keyphrases = [symbol, name]
    return keyphrases

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.stocks_file)
    df = df.nlargest(args.num_companies, "MarketCap")
    df["Keyphrases"] = df["Keyphrases"].fillna("")
    keyphrases_dict = {row["Symbol"]: row["Keyphrases"].split(", ") for _, row in df.T.iteritems()}
    datestrs = []
    datesfile = open(os.path.join(args.data_dir, "dates.txt"), "w")
    filenames_dict = defaultdict(lambda: defaultdict(list))
    for i, (root, dirs, filenames) in enumerate(os.walk(os.path.join(args.data_dir, "corpus"))):
        print(root)
        if not filenames: continue
        _, _, y, m, d = root.split("/")
        datestr = "-".join([y, m, d])
        for filename in filenames:
            with open(os.path.join(root, filename)) as f:
                text = f.read()
            for symbol, keyphrases in keyphrases_dict.items():
                regex = symbol if keyphrases[0] == "" else "|".join([symbol] + keyphrases)
                if re.search(regex, text):
                    filenames_dict[symbol][datestr].append(filename)
        datestrs.append(datestr)
    with open(os.path.join(args.data_dir, "dates.txt"), "w") as f_dates:
        for datestr in datestrs:
            f_dates.write(datestr + "\n")
    print(",".join(["date"] + sorted(list(filenames_dict.keys()))))
    for datestr in datestrs:
        print(",".join([datestr] + list(str(len(filenames_dict[symbol][datestr])) for symbol in sorted(filenames_dict))))
            
    for symbol in filenames_dict:
        if not os.path.isdir(os.path.join(args.data_dir, "signals", symbol)):
            os.makedirs(os.path.join(args.data_dir, "signals", symbol))
        with open(os.path.join(args.data_dir, "signals", symbol, "news.txt"), "w") as f_news:
            for datestr in datestrs:
                f_news.write(' '.join(filenames_dict[symbol][datestr]) + "\n")
