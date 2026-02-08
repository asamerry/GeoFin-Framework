import pandas as pd
import cvxpy as cp
import os, argparse, yaml
from glob import glob
import yfinance as yf

from models.markowitz import markowitz
from models.black_litterman import black_litterman

PENALTIES = {
    "none": lambda x: 0, 
    "l1": cp.norm1, 
    "l2": cp.sum_squares,
}

# Parse CLI Arguments
parser = argparse.ArgumentParser(usage="python3 main.py --config [config-file]")
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()

# Parse yaml config file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Load price data
if config["data-in"]["price-data"] == "live":
    print("Loading live data ...")
    assets = list(pd.read_csv("assets/asset-list.csv")["ABBREVIATION"])
    num_stocks = len(assets)
    data = [yf.Ticker(asset).history(period="5y", interval="1mo")[config["data-in"]["data-col"]] for asset in assets]
    prices = pd.DataFrame(dict(zip(assets, data)))
else:
    print(f"Loading data folder \"{config["data-in"]["price-data"]}\" ...")
    files = glob(f"{config["data-in"]["price-data"]}/*.csv")
    num_stocks = len(files)
    prices = pd.DataFrame(
        { os.path.basename(file).split(".")[0] : pd.read_csv(file, index_col=config["data-in"]["index-col"], date_format="%m/%Y")[config["data-in"]["data-col"]] for file in files }
    )
print(f"Data collected through {prices.index[-1].date()}\n")

# Call prescribed model
models = {"markowitz": markowitz, "black-litterman": black_litterman}
models[config["model"]["type"]](
    prices = prices, 
    portfolio_value = config["data-in"]["portfolio-value"], 
    short = config["model"]["short"], 
    confidence = config["model"]["confidence"],
    penalty = PENALTIES[config["model"]["penalty"]],
    penalty_weight = config["model"]["penalty-weight"],
    plot = config["data-out"]["plot"],
)