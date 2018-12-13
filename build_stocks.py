import argparse
import json
import numpy as np
import os
import pandas as pd
import requests
from datetime import datetime, timedelta

api_key = "Q38KZETZONEJ0L0H"
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir", type=str, default="data")
parser.add_argument("-f", "--dates_file", type=str, default="data/dates.txt")

def ingest_raw(symbol):
    response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}")
    d = json.loads(response.text)
    d = {datetime.strptime(datestr, "%Y-%m-%d"): float(signals["4. close"]) for datestr, signals in d["Time Series (Daily)"].items()}
    return list(d.keys()), list(d.values())

def linearly_interpolate(t, y, tstart, tend):
    s = pd.Series(y, index=t)
    s = s.resample('D')
    s = s.interpolate(method="linear")
    s = s.pct_change()
    s = s[tstart.strftime("%Y-%m-%d"): tend.strftime("%Y-%m-%d")]
    return s.astype(float).tolist()

def get_time_series(symbol, tstart, tend):
    t, y = ingest_raw("MSFT")
    return linearly_interpolate(t, y, datetime.strptime("2018-09-01", "%Y-%m-%d"), datetime.strptime("2018-10-01", "%Y-%m-%d"))

if __name__ == "__main__":
    args = parser.parse_args()
    symbols = os.listdir(os.path.join(args.data_dir, "signals"))
    with open(args.dates_file) as f:
        dates = [datetime.strptime(line[:-1], "%Y-%m-%d") for line in f]
    for symbol in symbols:
        print(symbol)
        stocks = get_time_series(symbol, dates[0], dates[-1])
        with open(os.path.join(args.data_dir, "signals", symbol, "stocks.txt"), "w") as f:
            for stock in stocks:
                f.write(f"{stock}\n")
