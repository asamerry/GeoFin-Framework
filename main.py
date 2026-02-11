import pandas as pd
import cvxpy as cp
import os, argparse, yaml
from datetime import datetime as dt
import yfinance as yf

from models.markowitz import MarkowitzModel
from models.capm import CAPModel
from models.black_litterman import BlackLittermanModel

PENALTIES = {
    "none": lambda x: 0, 
    "l1": cp.norm1, 
    "l2": cp.sum_squares,
}

# Parse CLI Arguments
parser = argparse.ArgumentParser(usage="python3 main.py --config [config-file] \nUse --recache to force new data.")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--recache", action="store_true", required=False)
args = parser.parse_args()

# Parse yaml config file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Load price data
date = dt.today().date()
prices_file = f"data/{date}-prices.csv"
if os.path.exists(prices_file) and not args.recache:
    print("Loading cached prices ...")
    prices = pd.read_csv(prices_file, index_col="Date")
    print(f"Data collected through {prices.index[-1].split(" ")[0]}")
else:
    print("Loading live prices ...")
    assets = list(pd.read_csv(config["data-in"]["assets-file"])["ABBREVIATION"])
    data = [yf.Ticker(asset).history(period=config["data-in"]["period"], interval=config["data-in"]["interval"])[config["data-in"]["data-col"]] for asset in assets]
    prices = pd.DataFrame(dict(zip(assets, data)))
    os.makedirs("data", exist_ok=True)
    prices.to_csv(prices_file)
    print(f"Data collected through {prices.index[-1].date()}")
rf_yearly = list(yf.Ticker("^TNX").history(period="1d", interval="1d")["Close"])[0] / 100
rf_monthly = (1 + rf_yearly) ** (1/12) - 1

# Call prescribed model
models = {"markowitz": MarkowitzModel, "capm": CAPModel, "black-litterman": BlackLittermanModel}
model = models[config["model"]["type"]](
    prices = prices, 
    portfolio_value = config["data-in"]["portfolio-value"], 
    short = config["model"]["short"], 
    penalty = PENALTIES[config["model"]["penalty"]],
    penalty_weight = config["model"]["penalty-weight"],
    rf = rf_monthly,
    views_file = config["data-in"]["views-file"], 
    recache = args.recache
)
model.solve()
model.print()

# Export data
if config["data-out"]["export"]:
    os.makedirs(config["data-out"]["export-file"].split("/")[0], exist_ok=True)
    with open(config["data-out"]["export-file"], "w") as file:
        file.write(model.portfolio.replace(", ", "\n"))
if config["data-out"]["plot"]: model.plot(config["data-out"]["export"], config["data-out"]["export-file"])