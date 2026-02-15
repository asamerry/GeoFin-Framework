import os, argparse, yaml

# Parse CLI Arguments
parser = argparse.ArgumentParser(usage="python3 main.py --config [config-file] -t [task] \nUse --recache to force new data.")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("-t", type=str, required=True)
parser.add_argument("--recache", action="store_true", required=False)
args = parser.parse_args()

# Parse yaml config file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Polymorphic main function
# Portfolio optimization
if args.t == "portfolio-optimization":
    from utils import get_prices_data
    from portfolio_optimization import MarkowitzOptimizer
    
    # Load price data
    prices, rf = get_prices_data(
        asset_file = config["data-in"]["asset-file"], 
        data_col = config["data-in"]["data-col"], 
        period = config["data-in"]["period"], 
        interval = config["data-in"]["interval"], 
        recache = args.recache
    )

    # Call prescribed optimization model
    optimizers = {"markowitz": MarkowitzOptimizer}
    optimizer = optimizers[config["portfolio-optimization"]["optimizer"]](
        prices = prices, 
        portfolio_value = config["data-in"]["portfolio-value"], 
        return_est = config["portfolio-optimization"]["returns"],
        risk_est = config["portfolio-optimization"]["risk"],
        short = config["portfolio-optimization"]["short"], 
        penalty = config["portfolio-optimization"]["penalty"],
        penalty_weight = config["portfolio-optimization"]["penalty-weight"],
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

# Options pricing
elif args.t == "options-pricing":
    import numpy as np
    from options_pricing import Option, BinomialPricer, BlackScholesPricer

    # Define options
    price_range = np.linspace(config["options-pricing"]["price-range"][0], config["options-pricing"]["price-range"][1], 10)
    vol_range = np.linspace(config["options-pricing"]["vol-range"][0], config["options-pricing"]["vol-range"][1], 10)
    options = [[Option(
        o_style = "european", 
        stock_p = price, 
        strike_p = config["options-pricing"]["strike-price"], 
        exp_t = config["options-pricing"]["time-to-expiry"], 
        sigma = vol, 
        int_r = config["options-pricing"]["risk-free-rate"]
    ) for price in price_range] for vol in vol_range]

    # Call prescribed pricing model
    pricers = {"binomial": BinomialPricer, "black-scholes": BlackScholesPricer}
    pricer = pricers[config["options-pricing"]["pricer"]]()
    pricer.heatmap(options)

    # Export data
    export_file = config["data-out"]["export-file"]
    if config["data-out"]["export"]: pass

# Invalid Task
else: print(f"Invalid task: {args.t}. Must be one of ['portfolio-optimization', 'options-pricing']")