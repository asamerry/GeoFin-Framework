import numpy as np
from numpy.linalg import inv
import pandas as pd
import yfinance as yf
from datetime import datetime as dt
import os, warnings
warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)

def get_prices_data(asset_file, data_col, period, interval, recache):
    date = dt.today().date()
    prices_file = f"data/{date}-prices.csv"
    if os.path.exists(prices_file) and not recache:
        print("Loading cached prices ...")
        prices = pd.read_csv(prices_file, index_col="Date")
        print(f"Data collected through {prices.index[-1].split(" ")[0]}")
    else:
        print("Loading live prices ...")
        assets = list(pd.read_csv(asset_file)["ABBREVIATION"])
        data = [yf.Ticker(asset).history(period=period, interval=interval)[data_col] for asset in assets]
        prices = pd.DataFrame(dict(zip(assets, data)))
        os.makedirs("data", exist_ok=True)
        prices.to_csv(prices_file)
        print(f"Data collected through {prices.index[-1].date()}")
    rf_yearly = list(yf.Ticker("^TNX").history(period="1d", interval="1d")["Close"])[0] / 100
    rf_monthly = (1 + rf_yearly) ** (1/12) - 1
    return prices, rf_monthly

def is_asset(tok, prices):
    return tok in prices.columns

def is_valid_view(view, prices):
    toks = view.split("; ")
    if len(toks) != 2: return False
    else: toks, conf = toks
    toks = toks.split()
    
    if not is_asset(toks[0], prices): return False 
    
    try: float(toks[1]); float(conf)
    except ValueError: return False
    
    if toks[2] not in ["up", "down", "over", "under"]:
        return False 
    elif toks[2] in ["up", "down"]:
        if len(toks) != 3 : return False
    else:
        if len(toks) != 4: return False
        if not is_asset(toks[3], prices): return False 

    return True

def parse_view(view, returns):
    toks, conf = view.split("; ")
    toks = toks.split(); conf = float(conf)
    if len(toks) == 3:
        dir = conf if toks[2] == "up" else -conf
        picks = [dir if toks[0] == abbr else 0 for abbr in returns.columns]
        return float(toks[1]), picks
    else:
        abbrs = [toks[0], toks[3]]
        dirs = [conf, -conf] if toks[2] == "over" else [-conf, conf]
        picks = []
        for abbr in returns.columns:
            if abbr == abbrs[0]: picks.append(dirs[0])
            elif abbr == abbrs[1]: picks.append(dirs[1])
            else: picks.append(0)
        return float(toks[1]), picks

def get_views(views_file, returns):
    Q = []; P = []

    with open(views_file, "r") as file:
        views = file.read().strip().split("\n")

    for view in views:
        if not is_valid_view(view, returns): 
            print(f"Invalid view statement: \"{view}\". Exiting.")
            exit(1)
        else:
            weight, picks = parse_view(view, returns)
            Q.append(weight)
            P.append(picks)

    return np.array(Q).reshape(-1, 1), np.array(P)

def get_market_weights(returns, recache):
    date = dt.today().date()
    market_caps_file = f"data/{date}-market-caps.csv"
    if os.path.exists(market_caps_file) and not recache:
        print("Loading cached market caps ...")
        market_caps = pd.read_csv(market_caps_file)["Market Cap"]
    else:
        print("Loading live market caps ...")
        assets = list(returns.columns)
        market_caps = {}
        for asset in assets:
            info = yf.Ticker(asset).info
            market_cap = info.get("marketCap", None)
            if market_cap is None:
                shares = info.get("sharesOutstanding", None)
                price = info.get("regularMarketPrice", None) or info.get("previousClose", None)
                if shares is not None and price is not None:
                    market_cap = shares * price
            if market_cap is None:
                market_cap = info.get("totalAssets", None)
            market_caps[asset] = market_cap
        
        market_caps = pd.Series(market_caps, name="Market Cap")
        market_caps.to_csv(market_caps_file)
        
    market_caps = market_caps.values.astype(float)
    w = market_caps / sum(market_caps)
    return np.array(w).reshape(-1, 1)

def get_returns(return_est, returns_historic, risk, rf, views_file, recache):
    if return_est == "capm":
        if views_file != "none":
            print(f"Unused parameter: {views_file=}")
        beta = np.array(risk.mean() / risk.values.sum())
        rm = returns_historic.mean()
        expected_returns = rf + beta * (rm - rf)
    elif return_est == "black-litterman":
        tau = 0.05 # [0, 1]
        Q, P = get_views(views_file, returns_historic)
        Omega = np.diag(np.diag(tau * P @ risk @ P.T))
        w_mkt = get_market_weights(returns_historic, recache)
        r_m = returns_historic.values @ w_mkt
        delta = r_m.mean() / r_m.var(ddof=1)
        Pi = delta * risk @ w_mkt
        expected_returns = inv(inv(tau * risk) + P.T @ inv(Omega) @ P) @ (inv(tau * risk) @ Pi + P.T @ inv(Omega) @ Q)
    else:
        if return_est != "historic": 
            print("Invalid return estimator. Defaulting to historic returns.")
        if views_file != "none":
            print(f"Unused parameter: {views_file=}")
        expected_returns = returns_historic[-12:].mean()
    return np.reshape(expected_returns, (-1, 1))

def get_risk(risk_est, returns_historic):
    if risk_est == "variance":
        risk = returns_historic.cov()
    else:
        print("Invalid risk estimator. Defaulting to variance risk.")
        risk = returns_historic.cov()
    return risk