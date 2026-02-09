import pandas as pd
import cvxpy as cp
import os, argparse, yaml
import yfinance as yf

from models.markowitz import MarkowitzModel
from models.black_litterman import BlackLittermanModel

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
print("Loading prices data ...")
assets = list(pd.read_csv(config["data-in"]["assets-file"])["ABBREVIATION"])
num_stocks = len(assets)
data = [yf.Ticker(asset).history(period=config["data-in"]["period"], interval=config["data-in"]["interval"])[config["data-in"]["data-col"]] for asset in assets]
prices = pd.DataFrame(dict(zip(assets, data)))
print(f"Data collected through {prices.index[-1].date()}\n")

# Call prescribed model
models = {"markowitz": MarkowitzModel, "black-litterman": BlackLittermanModel}
model = models[config["model"]["type"]](
    prices = prices, 
    portfolio_value = config["data-in"]["portfolio-value"], 
    short = config["model"]["short"], 
    penalty = PENALTIES[config["model"]["penalty"]],
    penalty_weight = config["model"]["penalty-weight"],
    views_file = config["data-in"]["views-file"]
)
model.solve()
model.print()
if config["data-out"]["plot"]: model.plot(config["data-out"]["export"], config["data-out"]["export-file"])

# Export data
if config["data-out"]["export"]:
    os.makedirs(config["data-out"]["export-file"].split("/")[0], exist_ok=True)
    with open(config["data-out"]["export-file"], "w") as file:
        file.write(model.portfolio.replace(", ", "\n"))