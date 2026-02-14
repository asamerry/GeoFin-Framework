import os, argparse, yaml

from utils import get_prices_data
from optimizers import MarkowitzOptimizer

# Parse CLI Arguments
parser = argparse.ArgumentParser(usage="python3 main.py --config [config-file] \nUse --recache to force new data.")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--recache", action="store_true", required=False)
args = parser.parse_args()

# Parse yaml config file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Load price data
prices, rf = get_prices_data(
    asset_file = config["data-in"]["asset-file"], 
    data_col = config["data-in"]["data-col"], 
    period = config["data-in"]["period"], 
    interval = config["data-in"]["interval"], 
    recache = args.recache
)

# Call prescribed model
optimizers = {"markowitz": MarkowitzOptimizer}
optimizer = optimizers[config["model"]["optimizer"]](
    prices = prices, 
    portfolio_value = config["data-in"]["portfolio-value"], 
    return_est = config["model"]["returns"],
    risk_est = config["model"]["risk"],
    short = config["model"]["short"], 
    penalty = config["model"]["penalty"],
    penalty_weight = config["model"]["penalty-weight"],
    rf = rf,
    views_file = config["data-in"]["views-file"], 
    recache = args.recache
)
optimizer.solve()
optimizer.print()

# Export data
export_file = config["data-out"]["export-file"]
if config["data-out"]["export"]:
    os.makedirs(export_file.split("/")[0], exist_ok=True)
    with open(export_file, "w") as file:
        file.write(optimizer.portfolio.replace(", ", "\n"))
elif export_file != "none":
    print(f"Unused parameter: {export_file=}")
if config["data-out"]["plot"]: optimizer.plot(config["data-out"]["export"], export_file)